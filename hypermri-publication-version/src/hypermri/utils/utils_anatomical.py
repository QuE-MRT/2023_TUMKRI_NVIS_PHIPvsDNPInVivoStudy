# Authors: Andre Wendlinger, andre.wendlinger@tum.de
#          Luca Nagel, luca.nagel@tum.de
#          Wolfgang Gottwald, wolfgang.gottwald@tum.de


# Institution: Technical University of Munich
# Date of last Edit (dd.mm.yyyy): 24.05.2023

from .logging import init_default_logger

import numpy as np
import ipywidgets as widgets
from skimage import measure
import matplotlib.pyplot as plt
from skimage.transform import resize
from mpl_interactions import image_segmenter
import os

logger = init_default_logger(__name__)

# see https://github.com/ianhi/mpl-interactions/pull/266
try:
    from mpl_interactions import image_segmenter_overlayed
except ImportError:
    logger.error("Image_segmenter_overlayed not found")


def bruker2complex(arr):
    """Turns bruker semi-complex arrays (2xfloat) into numpy.complex (1xcomplex)"""
    assert arr.shape[-1] == 2, "Last dimension must be complex dimension."
    return arr[..., 0] + 1j * arr[..., 1]


def get_segmenter_list_overlayed(
    anatomical_data,
    bssfp_data,
    anat_ext=None,
    bssfp_ext=None,
    overlay=0.25,
    bssfp_cmap="viridis",
    n_rois=1,
    figsize=(4, 4),
    vmin=None,
    vmax=None,
    mask_alpha=0.7,
):
    """
    Returns a list of image_segmenter_overlayed type entries. TO be read by draw_masks_on_anatomical
    Parameters
    ----------
    anatomical_data: np.array containing anatomical data
        shape: (read, phase, slice)
        Assumes that dim[2] is slice.
    bssfp_data: np.array containing bssfp data
        shape : (echos,read,phase,slice,reps,channels)
        Gets averaged over all reps per default. Can be changed later below.
    anat_ext : list, optional
        If bssfp and anatomical dont have the same extent you need to give this value.
        gives the extent of the anatomical image in mm for example: [-15,15,10,10]
    bssfp_ext : list, optional
        If bssfp and anatomical dont have the same extent you need to give this value.
        gives the extent of the bssfp image in mm for example: [-15,15,10,10]
    overlay : float
        alpha value of the secondary image, per default 0.25
    bssfp_cmap : str, optional
        bssfp colormap
    n_rois: int, optional
        number of rois to be segemented, so far limited to 1.
    Returns
    -------
    seg_list: list
        Contains image_segmenter_overlayed type objects
    """
    line_properties = {"color": "red", "linewidth": 1}
    seg_list = [
        image_segmenter_overlayed(
            anatomical_data[:, :, s],
            second_img=np.mean(bssfp_data[0, :, :, s, :, 0], axis=2),
            img_extent=anat_ext,
            second_img_extent=bssfp_ext,
            second_img_alpha=overlay,
            second_img_cmap=bssfp_cmap,
            figsize=figsize,
            nclasses=n_rois,
            props=line_properties,
            lineprops=None,
            mask_alpha=mask_alpha,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        for s in range(anatomical_data.shape[2])
    ]
    return seg_list


def draw_masks_on_anatomical(anatomical_segmenter_list, roi_names=None):
    """
    Loads a segmenter_list and then allows the user to draw ROIs which are saved in the segmenter_list
    """

    # define image plotting function
    def plot_imgs(n_slice, eraser_mode, roi_key):
        temp_seg = anatomical_segmenter_list[n_slice]
        temp_seg.erasing = eraser_mode
        if roi_names:
            # names instead of numbers
            roi_number = roi_names.index(roi_key) + 1
        else:
            # default numbering
            roi_number = roi_key
        temp_seg.current_class = roi_number
        display(temp_seg)

    n_rois = anatomical_segmenter_list[0].nclasses
    n_slices = len(anatomical_segmenter_list)

    # Making the UI
    if roi_names:
        class_selector = widgets.Dropdown(options=roi_names, description="ROI name")
    else:
        class_selector = widgets.Dropdown(
            options=list(range(1, n_rois + 1)), description="ROI number"
        )

    erasing_button = widgets.Checkbox(value=False, description="Erasing")
    # create interactive slider for echoes

    slice_slider = widgets.IntSlider(
        value=n_slices // 2, min=0, max=n_slices - 1, description="Slice: "
    )

    # put both sliders inside a HBox for nice alignment  etc.
    ui = widgets.HBox(
        [erasing_button, slice_slider, class_selector],
        layout=widgets.Layout(display="flex"),
    )

    sliders = widgets.interactive_output(
        plot_imgs,
        {
            "n_slice": slice_slider,
            "eraser_mode": erasing_button,
            "roi_key": class_selector,
        },
    )

    display(ui, sliders)


def get_masks(segmenter_list, roi_keys=None, plot_res=False):
    """
    Extract the masks for a given segmenter list.

    Parameters
    ---------
    segmenter_list: list of image_segmenter_overlayed objects

    roi_keys: list of str, optional
        Default is none, then just strings of numbers from 0-number_of_rois are the keys
        Suggested Roi key names: bloodvessel, tumor, kidneyL, kidneyR, tumor2, phantom, outside_ref

    plot_res: bool, optional.
        if one wants the result to be plotted for QA, default is False.
    Returns
    --------
    mask_per_slice: dict
        entries can be called via the selected keys and have
        shape: (read, phase, slice, number_of_rois)

    Examples
    --------
    If we give keys:
    masks = ut_anat.get_masks_multi_rois(segmenter_list,['Tumor','Kidney','Vessel'],True)
    masked_bssfp = masks['Tumor'] * bssfp_image

    If we dont give keys:
    masks = ut_anat.get_masks_multi_rois(segmenter_list)
    masks['1'].shape --> (read,phase,slice)
    masked_bssfp = masks['1'] * bssfp_image


    """
    n_slices = len(segmenter_list)
    n_rois = segmenter_list[0].nclasses
    if not roi_keys:
        # set default names
        roi_keys = [str(n) for n in range(1, n_rois + 1)]
    else:
        # use given keys
        pass
    mask_per_slice = np.zeros(
        (
            segmenter_list[0].mask.shape[0],
            segmenter_list[0].mask.shape[1],
            n_slices,
            n_rois,
        )
    )
    for slic in range(0, n_slices):
        for roi in range(0, n_rois):
            test_mask = segmenter_list[slic].mask == roi + 1
            mask_per_slice[:, :, slic, roi] = test_mask

    mask_dict = dict()

    for idx, roi_key in enumerate(roi_keys):
        mask_dict.update({roi_key: mask_per_slice[:, :, :, idx]})

    if plot_res:
        fig, ax = plt.subplots(1, n_rois)

        @widgets.interact(slices=(0, n_slices - 1, 1))
        def update(slices=0):
            if n_rois > 1:
                [ax[n].imshow(mask_per_slice[:, :, slices, n]) for n in range(n_rois)]
                [ax[n].set_title("ROI " + str(roi_keys[n])) for n in range(n_rois)]
            else:
                ax.imshow(mask_per_slice[:, :, slices, 0])
                ax.set_title("ROI " + str(roi_keys[0]))

        first_key = next(iter(mask_dict))
        for roi_num, key in enumerate(list(mask_dict.keys())):
            test = mask_dict[key]
            all_entries = []
            for slic in range(mask_dict[first_key].shape[2]):
                mask_entries = len(np.where(test[:, :, slic] > 0)[0])
                if mask_entries > 0:
                    all_entries.append(slic)
                else:
                    pass
            print(
                key,
                " is segmented in slices: ",
                all_entries,
                " of ",
                mask_dict[first_key].shape[2] - 1,
            )

    return mask_dict


def list_masks(dirpath):
    """
    Lists all files that end with masks.npz in a given directory.
    Parameters
    ----------
    dirpath: str
        Folder where we expect the mask files.

    Returns
    -------
    files: list of str
        contains names of all mask files in dirpath.
    """
    print("Found files in " + str(dirpath) + " :")
    files = []
    for file in os.listdir(dirpath):
        if file.endswith("masks.npz"):
            files.append(file)
            print(file)
    return files


def load_mask(dirpath, mask_name,plot_res = False):
    """
    Loads .npz file and retrieves the mask dictionary.
    Parameters
    ----------
    filepath: path to mask file

    Returns
    -------

    """
    data_loaded = np.load(os.path.join(dirpath, mask_name), allow_pickle=True)
    mask_dict = data_loaded["arr_0"][()]

    if plot_res:
        fig, ax = plt.subplots(1, n_rois)

        @widgets.interact(slices=(0, n_slices - 1, 1))
        def update(slices=0):

                [ax[n].imshow(mask_dict[key][:, :, slices]) for n in range(n_rois)]
                [ax[n].set_title("ROI " + str(roi_keys[n])) for n in range(n_rois)]
    return mask_dict


def get_roi_coords(mask_dict):
    """
    Extract the contours of a mask segmented with mpl_interactions image_segmenter_overlayed

    Parameters
    ---------
    segmenter_list: list

    Returns
    --------
    contours: list(np.arrays)
    """

    contours = list()

    first_key = next(iter(mask_dict))
    n_slices = mask_dict[first_key].shape[2]
    for roi in mask_dict.keys():
        for slic in range(n_slices):
            contours.append(measure.find_contours(mask_dict[roi][:, :, slic]))
    contours_reshaped = [
        contours[n : n + n_slices] for n in range(0, len(contours), n_slices)
    ]
    return contours_reshaped


def plot_segmented_roi_on_anat(
    axis, anatomical, mask_dict, slice_number, vmin_anat=False, vmax_anat=False
):
    """
    Plots the contours of a segmented mask on an anatomical image
    Parameters
    ----------
    axis: subplot object into which we plot.
    anatomical: np.array
    mask_dict: dict
        contains masks from segmentation
    slice: int
        Number of slice to plot.
    vmin_anat: float
        Windowing of anatomical image lower bound
    vmax_anat: float
        Windowing of anatomical image upper bound
    Returns
    -------

    """
    n_slices = anatomical.shape[2]
    try:
        roi_coords = get_roi_coords(mask_dict)
        if vmin_anat is not False:
            axis.imshow(
                anatomical[:, :, slice_number],
                cmap="gray",
                vmin=vmin_anat,
                vmax=vmax_anat,
            )
        else:
            axis.imshow(anatomical[:, :, slice_number], cmap="gray")
        for roi_num in range(len(roi_coords)):
            if len(roi_coords[roi_num][slice_number]) > 0:
                if len(roi_coords[roi_num][slice_number]) == 1:
                    axis.plot(
                        np.squeeze(roi_coords[roi_num][slice_number])[:, 1],
                        np.squeeze(roi_coords[roi_num][slice_number])[:, 0],
                        linewidth=2,
                        color="C" + str(roi_num),
                        label=list(mask_dict.keys())[roi_num],
                    )
                else:
                    for elem in range(len(roi_coords[roi_num][slice_number])):
                        axis.plot(
                            roi_coords[roi_num][slice_number][elem][:, 1],
                            roi_coords[roi_num][slice_number][elem][:, 0],
                            linewidth=2,
                            color="C" + str(roi_num),
                            alpha=1 / (elem + 1),
                            label=list(mask_dict.keys())[roi_num],
                        )

                # axis.set_title('Slice ' + str(slice_number))
        if len(axis.get_legend_handles_labels()[0])>0:
            axis.legend()
        else:
            axis.legend_ = None


    except IndexError:
        logger.critical("Slice number cannot be larger than " + str(n_slices))

def define_imagematrix_parameters(image_object):
    """
    Warning: Does not take into account the orientation and offsets of the
    object (yet)
    Define the imaging matrix in voxel.
    Returns imaging matrix dimensions  as dim_read, dim_phase, dim_slice
    Input
    -----
    image_object: Sequence object
    """
    if image_object is None:
        return None, None, None

    dim_read = image_object.method["PVM_Matrix"][0]  # was z
    dim_phase = image_object.method["PVM_Matrix"][1]  # was y
    if image_object.method["PVM_SpatDimEnum"] == "<3D>":
        dim_slice = image_object.method["PVM_Matrix"][2]  # was x
    else:
        dim_slice = image_object.method["PVM_SPackArrNSlices"]  # was x
    return dim_read, dim_phase, dim_slice


def define_imageFOV_parameters(image_object):
    """
    Warning: Does not take into account the orientation and offsets of the
    object (yet)
    Calculates the FOV in mm.
    Returns FOV in as mm_read, mm_phase, mm_slice.
    Input
    -----
    image_object: Sequence object
    """
    if image_object is None:
        return None, None, None

    # FOV:
    mm_read = image_object.method["PVM_Fov"][0]
    mm_phase = image_object.method["PVM_Fov"][1]
    mm_slice_gap = image_object.method["PVM_SPackArrSliceGap"]

    if image_object.method["PVM_SpatDimEnum"] == "<3D>":
        mm_slice = image_object.method["PVM_Fov"][2]
    else:
        _, _, dim_slice = define_imagematrix_parameters(image_object=image_object)
        mm_slice = image_object.method["PVM_SliceThick"]  # was x
        mm_slice = mm_slice * dim_slice + mm_slice_gap * (dim_slice - 1)

    return mm_read, mm_phase, mm_slice


def define_grid(image_object):
    """
    Warning: Does not take into account the orientation and offsets yet
    Defines a 2D/3D grid of the image.
    """

    if image_object is None:
        return None

    try:
        mat = np.array(define_imagematrix_parameters(image_object=image_object))
        fov = np.array(define_imageFOV_parameters(image_object=image_object))
    except:
        mat = image_object.method["PVM_Matrix"]
        fov = image_object.method["PVM_Fov"]

    # calculate resolution:
    res = fov / mat

    # init:
    ext_1 = ext_2 = ext_3 = None
    if (len(fov) > 0) and (len(mat) > 0):
        ext_1 = np.linspace(-fov[0] / 2 + res[0] / 2, fov[0] / 2 - res[0] / 2, mat[0])
    if (len(fov) > 1) and (len(mat) > 1):
        ext_2 = np.linspace(-fov[1] / 2 + res[1] / 2, fov[1] / 2 - res[1] / 2, mat[1])
    if (len(fov) > 2) and (len(mat) > 2):
        ext_3 = np.linspace(-fov[2] / 2 + res[2] / 2, fov[2] / 2 - res[2] / 2, mat[2])

    return ext_1, ext_2, ext_3

