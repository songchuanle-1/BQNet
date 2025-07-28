import numpy as np
import json
import open3d as o3d
import argparse
import os
import time

class SynchronizedVisualizer:
    def __init__(self):
        self.vis_list = []
        self.geometries = {}  # 存储每个可视化器中的几何对象
        self.viewpoint = None
        self.sync_viewpoint = False

    def add_point_cloud(self, pcd, window_name):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=window_name, width=400, height=400)
        vis.add_geometry(pcd)
        
        # 设置初始渲染选项
        render_opt = vis.get_render_option()
        render_opt.point_size = 2.0
        render_opt.background_color = np.array([1, 1, 1])
        
        # 存储几何对象引用
        self.geometries[vis] = pcd
        self.vis_list.append(vis)
        return vis

    def synchronize_view(self, source_vis):
        # 从源窗口获取当前视角
        ctr = source_vis.get_view_control()
        self.viewpoint = ctr.convert_to_pinhole_camera_parameters()
        
        # 应用到所有其他窗口
        for vis in self.vis_list:
            if vis != source_vis:
                target_ctr = vis.get_view_control()
                target_ctr.convert_from_pinhole_camera_parameters(self.viewpoint)
                # 更新几何体（如果需要）
                vis.update_geometry(self.geometries[vis])
                vis.poll_events()
                vis.update_renderer()

    def start_synchronization(self):
        self.sync_viewpoint = True
        print("视角同步已启用 - 请在任意窗口调整视角（调整第一个窗口为推荐）")

    def run(self, window_title_prefix="", save_path=None):
        if not self.vis_list:
            return
        
        print(f"使用{'同步' if self.sync_viewpoint else '独立'}视角运行可视化...")
        
        # 注册键盘回调
        def sync_callback(vis):
            self.synchronize_view(vis)
            print(f"视角已同步 ({window_title_prefix})")
            return False  # 表示继续运行
        
        for i, vis in enumerate(self.vis_list):
            # 设置窗口标题
            # vis.set_view_parameters(title=f"{window_title_prefix}_{i+1}")
            # if i == 0:
            #     vis.set_view_parameters(title=f"{window_title_prefix}_{i+1} (主控)")
            
            # 注册回调
            vis.register_key_callback(ord('S'), sync_callback)
            vis.register_key_callback(ord(' '), sync_callback)  # 空格键也可触发同步
            vis.register_key_callback(ord('C'), lambda vis: self.save_image(vis, save_path, f"{window_title_prefix}_{i}"))
        
        # 更新所有窗口
        active = True
        while active:
            active = False
            for vis in self.vis_list:
                try:
                    if vis.poll_events():
                        active = True
                        vis.update_renderer()
                        
                        # 如果同步启用，则应用当前视角
                        if self.sync_viewpoint and vis == self.vis_list[0]:
                            self.synchronize_view(vis)
                except RuntimeError:
                    pass
            
            time.sleep(0.01)
        
        # 关闭所有窗口
        for vis in self.vis_list:
            vis.destroy_window()

    def save_image(self, vis, save_path, suffix=""):
        if save_path is None:
            save_path = os.path.join(os.getcwd(), "screenshot")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 添加时间戳防止覆盖
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}_{suffix}_{timestamp}.png"
        
        try:
            vis.capture_screen_image(filename, do_render=True)
            print(f"已保存截图至: {filename}")
        except Exception as e:
            print(f"保存截图失败: {str(e)}")

def rle_decode(rle_dict):
    """从 RLE 字典解码回二进制掩码"""
    length = rle_dict['length']
    counts = list(map(int, rle_dict['counts'].split()))
    mask = np.zeros(length, dtype=np.bool_)
    for i in range(0, len(counts), 2):
        start = counts[i] - 1  # RLE 编码从1开始计数
        end = start + counts[i+1]
        mask[start:end] = True
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser('proceess_scanrefer')
    parser.add_argument('--compare_json_path', type=str, default='/sdc1/songcl/IPDN/code/IPDN/BQNet/BQNet_res/20250626_230823/results/BQNet_IPDN.json')
    parser.add_argument('--pc_root_path', type=str, default='/sdc1/songcl/IPDN/data/scannetv2/val')
    parser.add_argument('--gt_pmask_root_path', type=str, default='/sdc1/songcl/IPDN/code/IPDN/BQNet/BQNet_res/20250626_230823/results/gt_pmasks')
    parser.add_argument('--pre1_pmask_root_path', type=str, default='/sdc1/songcl/IPDN/code/IPDN/BQNet/BQNet_res/20250626_230823/results/pred_pmasks')
    parser.add_argument('--pre2_pmask_root_path', type=str, default='/sdc1/songcl/IPDN/code/IPDN/IPDN_checkpoints/model/results/pred_pmasks')
    parser.add_argument('--save_img_path', type=str, default='/sdc1/songcl/IPDN/code/IPDN/IPDN_checkpoints/model/results/compare_img')
    args = parser.parse_args()

    with open(args.compare_json_path, 'r') as f:
        kv = json.load(f)
    os.makedirs(args.save_img_path, exist_ok=True)

    for id in kv.keys():
        if id!="scene0671_00_[8]_26":continue
        # scene0663_00_[6]_73
        if kv[id]['BQNet']<kv[id]['IPDN']+0.5:continue
        print(f"\n处理点云 {id}")
        scene_names = id.split('_')
        pc_path = os.path.join(args.pc_root_path, scene_names[0] + '_' + scene_names[1] + '_vh_clean_2.ply')
        print(f"点云路径: {pc_path} ({os.path.exists(pc_path)})")
        
        gt_mask_path = os.path.join(args.gt_pmask_root_path, id + '.json')
        print(f"GT掩码路径: {gt_mask_path} ({os.path.exists(gt_mask_path)})")
        
        pred1_mask_path = os.path.join(args.pre1_pmask_root_path, id + '.json')
        print(f"预测掩码1路径: {pred1_mask_path} ({os.path.exists(pred1_mask_path)})")
        
        pred2_mask_path = os.path.join(args.pre2_pmask_root_path, id + '.json')
        print(f"预测掩码2路径: {pred2_mask_path} ({os.path.exists(pred2_mask_path)})")

        # 加载点云
        try:
            point_cloud = o3d.io.read_point_cloud(pc_path)
            if not point_cloud.has_points():
                raise ValueError("点云文件为空")
        except Exception as e:
            print(f"加载点云失败: {str(e)}")
            continue
        
        print(f"点云包含 {len(point_cloud.points)} 个点")
        
        # 加载掩码
        try:
            with open(gt_mask_path, 'r') as f:
                decoded_mask_gt = rle_decode(json.load(f))
            with open(pred1_mask_path, 'r') as f:
                decoded_mask_pre1 = rle_decode(json.load(f))
            with open(pred2_mask_path, 'r') as f:
                decoded_mask_pre2 = rle_decode(json.load(f))
        except Exception as e:
            print(f"加载掩码失败: {str(e)}")
            continue

        # 定义颜色
        bright_green = [0.0, 1.0, 0.0]
        bright_blue = [0.0, 0.0, 1.0]
        bright_red = [1.0, 0.0, 0.0]
        
        # 准备点云实例
        original_colors = np.asarray(point_cloud.colors)
        
        # 复制点云而不是共享点集
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = point_cloud.points
        pcd_original.colors = o3d.utility.Vector3dVector(original_colors)
        
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = point_cloud.points
        gt_colors = np.copy(original_colors)
        gt_colors[decoded_mask_gt] = bright_green
        pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)
        
        pcd_pre1 = o3d.geometry.PointCloud()
        pcd_pre1.points = point_cloud.points
        pre1_colors = np.copy(original_colors)
        pre1_colors[decoded_mask_pre1] = bright_blue
        pcd_pre1.colors = o3d.utility.Vector3dVector(pre1_colors)
        
        pcd_pre2 = o3d.geometry.PointCloud()
        pcd_pre2.points = point_cloud.points
        pre2_colors = np.copy(original_colors)
        pre2_colors[decoded_mask_pre2] = bright_red
        pcd_pre2.colors = o3d.utility.Vector3dVector(pre2_colors)

        # 创建同步可视化器
        sync_vis = SynchronizedVisualizer()
        
        # 添加点云到可视化器
        sync_vis.add_point_cloud(pcd_original, "原始点云")
        sync_vis.add_point_cloud(pcd_gt, "GT掩码点云")
        sync_vis.add_point_cloud(pcd_pre1, "预测掩码1点云")
        sync_vis.add_point_cloud(pcd_pre2, "预测掩码2点云")
        
        # 设置截图路径
        save_img_path = os.path.join(args.save_img_path, id)
        
        # 打印使用说明
        print("操作说明:")
        print("  1. 按 'S' 或 空格键 同步所有窗口的视角")
        print("  2. 按 'C' 在任意窗口捕获当前视图的截图")
        print("  3. 关闭所有窗口进入下一个点云")
        
        # 启动同步视图
        sync_vis.sync_viewpoint = True  # 开始就启用同步
        
        # 运行可视化
        sync_vis.run(window_title_prefix=id, save_path=save_img_path)
        
        print(f"已处理点云 {id}，准备下一个...\n")