# 安全模式：先停止所有GPU进程
sudo pkill -f cuda
sudo pkill -f nvidia

# 卸载NVIDIA驱动模块
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia

# 重新加载NVIDIA驱动模块
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_drm
sudo modprobe nvidia_modeset

# 检查是否恢复
nvidia-smi
