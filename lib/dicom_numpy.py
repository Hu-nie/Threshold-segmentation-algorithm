import logging
from math import isclose

import numpy as np

from lib.exceptions import DicomImportException, MissingInstanceNumberException


logger = logging.getLogger(__name__)


def combine_slices(datasets, rescale=None, enforce_slice_spacing=True, sort_by_instance=False):

    slice_datasets = [ds for ds in datasets if not _is_dicomdir(ds)]

    if len(slice_datasets) == 0:
        raise DicomImportException("Must provide at least one image DICOM dataset")

    if sort_by_instance:
        sorted_datasets = sort_by_instance_number(slice_datasets)
    else:
        sorted_datasets = sort_by_slice_position(slice_datasets)

    _validate_slices_form_uniform_grid(sorted_datasets, enforce_slice_spacing=enforce_slice_spacing)

    voxels = _merge_slice_pixel_arrays(sorted_datasets, rescale)
    transform = _ijk_to_patient_xyz_transform_matrix(sorted_datasets)

    return voxels, transform


def sort_by_instance_number(slice_datasets):

    instance_numbers = [getattr(ds, 'InstanceNumber', None) for ds in slice_datasets]
    if any(n is None for n in instance_numbers):
        raise MissingInstanceNumberException

    return [
        d for (s, d) in sorted(
            zip(instance_numbers, slice_datasets),
            key=lambda v: int(v[0]),
            # Stacked in reverse to order in direction of increasing slice axis
            reverse=True
        )
    ]


def sort_by_slice_position(slice_datasets):

    slice_positions = _slice_positions(slice_datasets)
    return [
        d for (s, d) in sorted(
            zip(slice_positions, slice_datasets),
            key=lambda v: v[0],
        )
    ]


def _is_dicomdir(dataset):
    media_sop_class = getattr(dataset, 'MediaStorageSOPClassUID', None)
    return media_sop_class == '1.2.840.10008.1.3.10'


def _merge_slice_pixel_arrays(sorted_datasets, rescale=None):
    if rescale is None:
        rescale = any(_requires_rescaling(d) for d in sorted_datasets)

    first_dataset = sorted_datasets[0]
    # print(first_dataset)
    slice_dtype = first_dataset.pixel_array.dtype
    slice_shape = first_dataset.pixel_array.T.shape
    num_slices = len(sorted_datasets)

    voxels_shape = slice_shape + (num_slices,)
    voxels_dtype = np.float32 if rescale else slice_dtype
    voxels = np.empty(voxels_shape, dtype=voxels_dtype, order='F')
    
    if rescale:
        try: 
            intercept = first_dataset[0x20051409].value
            slope = first_dataset[0x2005140A].value
        except KeyError:
            print('rescale slope and intercept does not exist')
            intercept = 0
            slope = 1

    for k, dataset in enumerate(sorted_datasets):
        pixel_array = dataset.pixel_array.T
        # slope = float(getattr(dataset, 'RescaleSlope', 1))
        # intercept = float(getattr(dataset, 'RescaleIntercept', 0))
        pixel_array = pixel_array.astype(np.float32) * slope + intercept
        voxels[..., k] = pixel_array

    return voxels


def _requires_rescaling(dataset):
    return hasattr(dataset, 'RescaleSlope') or hasattr(dataset, 'RescaleIntercept')


def _ijk_to_patient_xyz_transform_matrix(sorted_datasets):
    first_dataset = sorted_datasets[0]
    image_orientation = first_dataset.ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    row_spacing, column_spacing = first_dataset.PixelSpacing
    slice_spacing = _slice_spacing(sorted_datasets)

    transform = np.identity(4, dtype=np.float32)

    transform[:3, 0] = row_cosine * column_spacing
    transform[:3, 1] = column_cosine * row_spacing
    transform[:3, 2] = slice_cosine * slice_spacing

    transform[:3, 3] = first_dataset.ImagePositionPatient

    return transform


def _validate_slices_form_uniform_grid(sorted_datasets, enforce_slice_spacing=True):

    invariant_properties = [
        'Modality',
        'SOPClassUID',
        'SeriesInstanceUID',
        'Rows',
        'Columns',
        'SamplesPerPixel',
        'PixelSpacing',
        'PixelRepresentation',
        'BitsAllocated',
    ]

    for property_name in invariant_properties:
        _slice_attribute_equal(sorted_datasets, property_name)

    _validate_image_orientation(sorted_datasets[0].ImageOrientationPatient)
    _slice_ndarray_attribute_almost_equal(sorted_datasets, 'ImageOrientationPatient', 1e-5)

    if enforce_slice_spacing:
        slice_positions = _slice_positions(sorted_datasets)
        _check_for_missing_slices(slice_positions)


def _validate_image_orientation(image_orientation):

    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    if not _almost_zero(np.dot(row_cosine, column_cosine), 1e-4):
        raise DicomImportException(f"Non-orthogonal direction cosines: {row_cosine}, {column_cosine}")
    elif not _almost_zero(np.dot(row_cosine, column_cosine), 1e-8):
        logger.warning(f"Direction cosines aren't quite orthogonal: {row_cosine}, {column_cosine}")

    if not _almost_one(np.linalg.norm(row_cosine), 1e-4):
        raise DicomImportException(f"The row direction cosine's magnitude is not 1: {row_cosine}")
    elif not _almost_one(np.linalg.norm(row_cosine), 1e-8):
        logger.warning(f"The row direction cosine's magnitude is not quite 1: {row_cosine}")

    if not _almost_one(np.linalg.norm(column_cosine), 1e-4):
        raise DicomImportException(f"The column direction cosine's magnitude is not 1: {column_cosine}")
    elif not _almost_one(np.linalg.norm(column_cosine), 1e-8):
        logger.warning(f"The column direction cosine's magnitude is not quite 1: {column_cosine}")


def _almost_zero(value, abs_tol):
    return isclose(value, 0.0, abs_tol=abs_tol)


def _almost_one(value, abs_tol):
    return isclose(value, 1.0, abs_tol=abs_tol)


def _extract_cosines(image_orientation):
    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)
    return row_cosine, column_cosine, slice_cosine


def _slice_attribute_equal(sorted_datasets, property_name):
    initial_value = getattr(sorted_datasets[0], property_name, None)
    for dataset in sorted_datasets[1:]:
        value = getattr(dataset, property_name, None)
        if value != initial_value:
            msg = f'All slices must have the same value for "{property_name}": {value} != {initial_value}'
            raise DicomImportException(msg)


def _slice_ndarray_attribute_almost_equal(sorted_datasets, property_name, abs_tol):
    initial_value = getattr(sorted_datasets[0], property_name, None)
    for dataset in sorted_datasets[1:]:
        value = getattr(dataset, property_name, None)
        if not np.allclose(value, initial_value, atol=abs_tol):
            msg = (f'All slices must have the same value for "{property_name}" within "{abs_tol}": {value} != '
                   f'{initial_value}')
            raise DicomImportException(msg)


def _slice_positions(sorted_datasets):
    image_orientation = sorted_datasets[0].ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)
    return [np.dot(slice_cosine, d.ImagePositionPatient) for d in sorted_datasets]


def _check_for_missing_slices(slice_positions):
    if len(slice_positions) > 1:
        slice_positions_diffs = np.diff(sorted(slice_positions))
        if not np.allclose(slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-5):
            # TODO: figure out how we should handle non-even slice spacing
            msg = f"The slice spacing is non-uniform. Slice spacings:\n{slice_positions_diffs}"
            logger.warning(msg)

        if not np.allclose(slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-1):
            raise DicomImportException('It appears there are missing slices')


def _slice_spacing(sorted_datasets):
    if len(sorted_datasets) > 1:
        slice_positions = _slice_positions(sorted_datasets)
        slice_positions_diffs = np.diff(slice_positions)
        return np.median(slice_positions_diffs)

    return getattr(sorted_datasets[0], 'SpacingBetweenSlices', 0)