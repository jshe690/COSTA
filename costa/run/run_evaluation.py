#   Edits to this code made by Jiantao Shen, 20.05.25

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import surface_distance
import SimpleITK as sitk
from prettytable import PrettyTable
import os
from skimage.morphology import skeletonize_3d, skeletonize
from skimage.measure import marching_cubes, find_contours, mesh_surface_area
import pandas as pd
import argparse
from rich.progress import track
from rich import print


def _find_all_files(path, model=None):
    """
    :param path: [str] path to the folder
    :param extension: [str] file extension
    :return: [list]
    """
    if model == '3DUNet':
        all_image_files = []
        for file in os.listdir(path):
            if file.endswith("_seg.nii.gz"):
                if 'logits' not in file:
                    all_image_files.append(os.path.join(path, file))
    else:
        all_image_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".nii.gz"):
                    if 'logits' not in file:
                        all_image_files.append(os.path.join(root, file))
    return all_image_files


def cl_score(v, s):
    return np.sum(v * s) / np.sum(s)


def compute_cl_dice(v_p, v_l):
    """
    :param v_p: [bool] predicted image
    :param v_l: [bool] ground truth
    :return: [floot] clDice
    """
    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif len(v_p.shape) == 3:
        tprec = cl_score(v_p, skeletonize_3d(v_l))
        tsens = cl_score(v_l, skeletonize_3d(v_p))
    cldice = 2 * tprec * tsens / (tprec + tsens)
    return cldice

def compute_del_V_SA_SAV(v_p, v_l, spacing):
    """
    :param v_p: [bool] predicted image
    :param v_l: [bool] ground truth
    :return: [floot] clDice
    """

    def surface_area_3d(segmentation, spacing):
        verts, faces, _, _ = marching_cubes(segmentation, level=None, spacing=spacing)
        SA = mesh_surface_area(verts, faces)
        return SA

    vol_p = np.sum(v_p) * np.prod(spacing)
    vol_l = np.sum(v_l) * np.prod(spacing)
    delV = vol_p / vol_l

    SA_p = surface_area_3d(v_p, spacing)
    SA_l = surface_area_3d(v_l, spacing)
    delSA = SA_p / SA_l

    SAV_p = SA_p / vol_p
    SAV_l = SA_l / vol_l
    del_SAV = SAV_p / SAV_l

    return delV, delSA, del_SAV

def get_label_using_model_selection(gt_path, file, instance_name, model):
    # tag = os.path.basename(file)[0:20]
    # label_path = os.path.join(gt_path, tag + '_seg.nii')
    print(file)
    label_path = os.path.join(gt_path, os.path.basename(file))
    label = sitk.ReadImage(label_path)

    return label

def evaluation(pred_path, gt_path, model, measure_type=None):
    if measure_type is None:
        # measure_type = ['avg', 'hd95', 'dice', 'surf-dice', 'cldice', 'delta-volume', 'delta-SA', 'delta-SAV-ratio']
        measure_type = ['dice', 'hd95', 'avg', 'surf-dice', 'cldice', 'delta-volume', 'delta-SA', 'delta-SAV-ratio']

    avg_distances, hd95_scores, dice_scores, surf_dice_scores, cldice_scores, delV_scores, delSA_scores, delSAV_scores = [], [], [], [], [], [], [], []
    image_names = []
    all_measures = []
    quantified_results = OrderedDict()

    all_images = _find_all_files(pred_path, model)
    print(all_images)
    if len(all_images) == 0:
        raise ValueError('No images found in {}'.format(pred_path))
    for file in track(all_images, description=f"Evaluating... {model}", total=len(all_images)):
        instance_name = os.path.basename(file)[:-7]
        image_names.append(instance_name)
        pred = sitk.ReadImage(file)
        label = get_label_using_model_selection(gt_path, file, instance_name, model)
        spacing = pred.GetSpacing()
        spacing = spacing[::-1]

        pred = sitk.GetArrayFromImage(pred)
        label = sitk.GetArrayFromImage(label)

        ############################################################################################
        # if model == '3DUNet':
        #     pred = pred[2:-2, 2:-2, 2:-2]
        #     label = label[2:-2, 2:-2, 2:-2]
        #     print('Warning, cropping the images for evaluation for 3DUNet!!!')
        # print(pred.shape)
        # print(label.shape)
        # print(spacing)
        ############################################################################################

        # normalize to [0, 1]
        pred[pred > 0] = 1
        label[label > 0] = 1

        surf_distance = surface_distance.compute_surface_distances(label.astype(bool), pred.astype(bool),
                                                                   spacing_mm=spacing)

        current_measures = []
        if 'avg' in measure_type:
            avg_surf_dist = surface_distance.compute_average_surface_distance(surf_distance)
            avg_surf_dist = np.average(avg_surf_dist)
            avg_distances.append(avg_surf_dist)
            current_measures.append(avg_surf_dist)

        if 'hd95' in measure_type:
            hd = surface_distance.compute_robust_hausdorff(surf_distance, 95)
            hd95_scores.append(hd)
            current_measures.append(hd)

        if 'dice' in measure_type:
            dice = surface_distance.compute_dice_coefficient(label.astype(np.uint8), pred.astype(np.uint8))
            dice_scores.append(dice)
            current_measures.append(dice)

        if 'surf-dice' in measure_type:
            # print('sasasasasas')
            # print('sasasasasas')
            # print('sasasasasas')
            # print('sasasasasas')
            # print('sasasasasas')
            # print('sasasasasas')
            # print('sasasasasas')
            # print('sasasasasas')
            # print('sasasasasas')
            sdice = surface_distance.compute_surface_dice_at_tolerance(surf_distance, tolerance_mm=spacing[0])
            # sdice = surface_distance.compute_surface_dice_at_tolerance(surf_distance, tolerance_mm=0)
            surf_dice_scores.append(sdice)
            current_measures.append(sdice)

        if 'cldice' in measure_type:
            cldice = compute_cl_dice(pred.astype(np.uint8), label.astype(np.uint8))
            cldice_scores.append(cldice)
            current_measures.append(cldice)

        if 'delta-volume' in measure_type:
            assert 'delta-SA' in measure_type, 'xxx'
            assert 'delta-SAV-ratio' in measure_type, 'xxx'
            delV, delSA, del_SAV = compute_del_V_SA_SAV(pred.astype(np.uint8), label.astype(np.uint8), spacing)
            delV_scores.append(delV)
            delSA_scores.append(delSA)
            delSAV_scores.append(del_SAV)
            current_measures.append(delV)
            current_measures.append(delSA)
            current_measures.append(del_SAV)

        all_measures.append(current_measures)

    if 'avg' in measure_type:
        quantified_results["avg"] = [np.mean(np.array(avg_distances)), np.std(np.array(avg_distances))]

    if 'hd95' in measure_type:
        quantified_results["hd95"] = [np.mean(np.array(hd95_scores)), np.std(np.array(hd95_scores))]

    if 'dice' in measure_type:
        quantified_results["dice"] = [np.mean(np.array(dice_scores)), np.std(np.array(dice_scores))]

    if 'surf-dice' in measure_type:
        quantified_results["surf-dice"] = [np.mean(np.array(surf_dice_scores)), np.std(np.array(surf_dice_scores))]

    if 'cldice' in measure_type:
        quantified_results["cldice"] = [np.mean(np.array(cldice_scores)), np.std(np.array(cldice_scores))]

    if 'delta-volume' in measure_type:
        assert 'delta-SA' in measure_type, 'xxx'
        assert 'delta-SAV-ratio' in measure_type, 'xxx'
        quantified_results["delta-volume"] = [np.mean(np.array(delV_scores)), np.std(np.array(delV_scores))]
        quantified_results["delta-SA"] = [np.mean(np.array(delSA_scores)), np.std(np.array(delSA_scores))]
        quantified_results["delta-SAV-ratio"] = [np.mean(np.array(delSAV_scores)), np.std(np.array(delSAV_scores))]

    return quantified_results, all_measures, image_names

def evaluation_original(pred_path, gt_path, measure_type=None):
    if measure_type is None:
        measure_type = ['avg', 'hd95', 'dice', 'surf-dice', 'cldice']

    avg_distances, hd95_scores, dice_scores, surf_dice_scores, cldice_scores = [], [], [], [], []
    image_names = []
    all_measures = []
    quantified_results = OrderedDict()

    all_images = _find_all_files(pred_path)
    if len(all_images) == 0:
        raise ValueError('No images found in {}'.format(pred_path))
    for file in track(all_images, description="Evaluating...", total=len(all_images)):
        instance_name = os.path.basename(file)[:-7]
        image_names.append(instance_name)
        pred = sitk.ReadImage(file)
        label = sitk.ReadImage(os.path.join(gt_path, os.path.basename(file)))
        spacing = pred.GetSpacing()
        spacing = spacing[::-1]

        pred = sitk.GetArrayFromImage(pred)
        label = sitk.GetArrayFromImage(label)

        # normalize to [0, 1]
        pred[pred > 0] = 1
        label[label > 0] = 1

        surf_distance = surface_distance.compute_surface_distances(label.astype(bool), pred.astype(bool),
                                                                   spacing_mm=spacing)

        current_measures = []
        if 'avg' in measure_type:
            avg_surf_dist = surface_distance.compute_average_surface_distance(surf_distance)
            avg_surf_dist = np.average(avg_surf_dist)
            avg_distances.append(avg_surf_dist)
            current_measures.append(avg_surf_dist)

        if 'hd95' in measure_type:
            hd = surface_distance.compute_robust_hausdorff(surf_distance, 95)
            hd95_scores.append(hd)
            current_measures.append(hd)

        if 'dice' in measure_type:
            dice = surface_distance.compute_dice_coefficient(label.astype(np.uint8), pred.astype(np.uint8))
            dice_scores.append(dice)
            current_measures.append(dice)

        if 'surf-dice' in measure_type:
            sdice = surface_distance.compute_surface_dice_at_tolerance(surf_distance, tolerance_mm=spacing[0])
            surf_dice_scores.append(sdice)
            current_measures.append(sdice)

        if 'cldice' in measure_type:
            cldice = compute_cl_dice(pred.astype(np.uint8), label.astype(np.uint8))
            cldice_scores.append(cldice)
            current_measures.append(cldice)

        all_measures.append(current_measures)

    if 'avg' in measure_type:
        quantified_results["avg"] = [np.mean(np.array(avg_distances)), np.std(np.array(avg_distances))]

    if 'hd95' in measure_type:
        quantified_results["hd95"] = [np.mean(np.array(hd95_scores)), np.std(np.array(hd95_scores))]

    if 'dice' in measure_type:
        quantified_results["dice"] = [np.mean(np.array(dice_scores)), np.std(np.array(dice_scores))]

    if 'surf-dice' in measure_type:
        quantified_results["surf-dice"] = [np.mean(np.array(surf_dice_scores)), np.std(np.array(surf_dice_scores))]

    if 'cldice' in measure_type:
        quantified_results["cldice"] = [np.mean(np.array(cldice_scores)), np.std(np.array(cldice_scores))]

    return quantified_results, all_measures, image_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pred_path", type=str, help="path to the predicted images", required=True)
    parser.add_argument("-g", "--gt_path", type=str, help="path to the ground truth images", required=True)
    parser.add_argument("-avg", "--average_surface_distance", type=bool, help="measure type", default=True)
    parser.add_argument("-hd95", "--hausdorff_distance_95", type=bool, help="measure type", default=True)
    parser.add_argument("-dice", "--dice_coefficient", type=bool, help="measure type", default=True)
    parser.add_argument("-sd", "--surface_dice", type=bool, help="measure type", default=True)
    parser.add_argument("-cldice", "--cl_dice", type=bool, help="measure type", default=True)
    parser.add_argument("-id", "--image_id", type=str, help="image id", default=None)

    parser.add_argument("-delV", "--delta_volume", type=bool, help="measure type", default=True)
    parser.add_argument("-delSA", "--delta_surface_area", type=bool, help="measure type", default=True)
    parser.add_argument("-delSAVr", "--delta_SA_V_ratio", type=bool, help="measure type", default=True)
    parser.add_argument("-m", "--model", type=str, help="3DUNet, nnUNet, COSTA", required=True, default=None)

    args = parser.parse_args()
    pred_path = args.pred_path
    gt_path = args.gt_path
    img_id = args.image_id
    model = args.model

    measure_type = []
    column_names = []
    if args.dice_coefficient:
        measure_type.append('dice')
        column_names.append('DICE')
    if args.hausdorff_distance_95:
        measure_type.append('hd95')
        column_names.append('HD95')
    if args.average_surface_distance:
        measure_type.append('avg')
        column_names.append('ASD')
    if args.surface_dice:
        measure_type.append('surf-dice')
        column_names.append('SurfDice')
    if args.cl_dice:
        measure_type.append('cldice')
        column_names.append('clDice')
    if args.delta_volume:
        measure_type.append('delta-volume')
        column_names.append('delta-volume')
    if args.delta_surface_area:
        measure_type.append('delta-SA')
        column_names.append('delta-SA')
    if args.delta_SA_V_ratio:
        measure_type.append('delta-SAV-ratio')
        column_names.append('delta-SAV-ratio')

    print("\nEvaluation started...")
    quantified_results, all_measures, image_names = evaluation(pred_path, gt_path, model, measure_type)

    # save the results to excel file
    abs_pred_path = os.path.abspath(pred_path)
    df = pd.DataFrame(all_measures, columns=column_names)
    df.insert(0, 'Image', image_names)
    # insert id
    img_ids = [img_id for i in range(len(image_names))]
    if img_id is not None:
        df.insert(1, 'ID', img_ids)
    df.to_excel(os.path.join(os.path.dirname(abs_pred_path), os.path.basename(abs_pred_path) + "_eval_results.xlsx"),
                index=False)
    print("Results saved to:",
          os.path.join(os.path.dirname(abs_pred_path), os.path.basename(abs_pred_path) + "_eval_results.xlsx"))

    # some info to write
    rows = []
    latex_str = ''
    for m in measure_type:
        ret = quantified_results[m]
        ret_formatted = '%.3f' % ret[0] + "±" + '%.3f' % ret[1]
        rows.append(ret_formatted)
        latex_str = f'{latex_str} ${ret[0]:.3f} \pm {ret[1]:.3f}$ &'
    table = PrettyTable()
    table.field_names = column_names
    table.add_row(rows)

    print("\nOverall performance:")
    with open(
            os.path.join(os.path.dirname(abs_pred_path), os.path.basename(abs_pred_path) + "_overall_performance.txt"),
            "w+",
            encoding="utf-8") as f:
        f.write(table.get_string())
        f.write(latex_str)
    print(table)
    print(latex_str)
def main_original():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pred_path", type=str, help="path to the predicted images", required=True)
    parser.add_argument("-g", "--gt_path", type=str, help="path to the ground truth images", required=True)
    parser.add_argument("-avg", "--average_surface_distance", type=bool, help="measure type", default=True)
    parser.add_argument("-hd95", "--hausdorff_distance_95", type=bool, help="measure type", default=True)
    parser.add_argument("-dice", "--dice_coefficient", type=bool, help="measure type", default=True)
    parser.add_argument("-sd", "--surface_dice", type=bool, help="measure type", default=True)
    parser.add_argument("-cldice", "--cl_dice", type=bool, help="measure type", default=True)
    parser.add_argument("-id", "--image_id", type=str, help="image id", default=None)

    args = parser.parse_args()
    pred_path = args.pred_path
    gt_path = args.gt_path
    img_id = args.image_id

    measure_type = []
    column_names = []
    if args.average_surface_distance:
        measure_type.append('avg')
        column_names.append('ASD')
    if args.hausdorff_distance_95:
        measure_type.append('hd95')
        column_names.append('HD95')
    if args.dice_coefficient:
        measure_type.append('dice')
        column_names.append('DICE')
    if args.surface_dice:
        measure_type.append('surf-dice')
        column_names.append('SurfDice')
    if args.cl_dice:
        measure_type.append('cldice')
        column_names.append('clDice')

    print("\nEvaluation started...")
    quantified_results, all_measures, image_names = evaluation(pred_path, gt_path, measure_type)

    # save the results to excel file
    abs_pred_path = os.path.abspath(pred_path)
    df = pd.DataFrame(all_measures, columns=column_names)
    df.insert(0, 'Image', image_names)
    # insert id
    img_ids = [img_id for i in range(len(image_names))]
    if img_id is not None:
        df.insert(1, 'ID', img_ids)
    df.to_excel(os.path.join(os.path.dirname(abs_pred_path), os.path.basename(abs_pred_path) + "_eval_results.xlsx"),
                index=False)
    print("Results saved to:",
          os.path.join(os.path.dirname(abs_pred_path), os.path.basename(abs_pred_path) + "_eval_results.xlsx"))

    # some info to write
    rows = []
    for m in measure_type:
        ret = quantified_results[m]
        ret = '%.3f' % ret[0] + "±" + '%.3f' % ret[1]
        rows.append(ret)
    table = PrettyTable()
    table.field_names = column_names
    table.add_row(rows)

    print("\nOverall performance:")
    with open(
            os.path.join(os.path.dirname(abs_pred_path), os.path.basename(abs_pred_path) + "_overall_performance.txt"),
            "w+",
            encoding="utf-8") as f:
        f.write(table.get_string())
    print(table)


if __name__ == '__main__':
    main()
