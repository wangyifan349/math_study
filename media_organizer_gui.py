import os
import shutil
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from datetime import datetime
from collections import defaultdict

# 定义支持的文件扩展名
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.heic', '.webp'
}
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.mts', '.m2ts', '.ts'
}
AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.aac', '.flac', '.ogg', '.wma', '.m4a', '.ape', '.alac'
}

# ----------------------------------------------------------------------------------------------------------------------
class MediaOrganizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("媒体文件整理工具")
        self.root.geometry("700x500")
        self.root.resizable(False, False)
        self.source_directories = []      # 存储源目录列表
        self.destination_directory = ""   # 存储目标目录
        self.move_files = tk.BooleanVar()
        self.move_files.set(False)        # 默认不移动文件，而是复制
        self.progress = 0

        self.setup_styles()       # 设置界面样式
        self.create_widgets()     # 创建界面控件

    # ------------------------------------------------------------------------------------------------------------------
    def setup_styles(self):
        """
        设置全局样式，包括字体和背景颜色等
        """
        style = ttk.Style()

        # 配置按钮样式
        style.configure("TButton", font=("Microsoft YaHei", 10))

        # 配置标签样式
        style.configure("TLabel", font=("Microsoft YaHei", 10))

        # 配置复选框样式
        style.configure("TCheckbutton", font=("Microsoft YaHei", 10))

        # 配置输入框样式
        style.configure("TEntry", font=("Microsoft YaHei", 10))

        # 配置框架背景色
        style.configure("TFrame", background="#F5F5F5")
        style.configure("TLabelFrame", background="#F5F5F5")

        # 配置进度条样式
        style.configure("Horizontal.TProgressbar", thickness=20)

    # ------------------------------------------------------------------------------------------------------------------
    def create_widgets(self):
        """
        创建界面上的所有控件，包括源目录、目标目录、选项、操作按钮、进度条和日志
        """
        # 源目录选择框架
        source_frame = ttk.LabelFrame(
            self.root, text=" 源目录 ", padding=(10, 5)
        )
        source_frame.place(x=10, y=10, width=680, height=150)

        # 源目录列表框
        self.source_listbox = tk.Listbox(
            source_frame, height=5, font=("Microsoft YaHei", 10)
        )
        self.source_listbox.place(x=10, y=10, width=530, height=120)

        # 源目录操作按钮框架
        source_button_frame = ttk.Frame(source_frame)
        source_button_frame.place(x=550, y=10, width=110, height=120)

        # 添加目录按钮
        add_source_btn = ttk.Button(
            source_button_frame, text="添加目录", command=self.add_source_directory
        )
        add_source_btn.pack(fill="x", padx=5, pady=5)

        # 移除选中目录按钮
        remove_source_btn = ttk.Button(
            source_button_frame, text="移除选中", command=self.remove_selected_source
        )
        remove_source_btn.pack(fill="x", padx=5, pady=5)

        # 清空列表按钮
        clear_source_btn = ttk.Button(
            source_button_frame, text="清空列表", command=self.clear_source_directories
        )
        clear_source_btn.pack(fill="x", padx=5, pady=5)

        # 目标目录选择框架
        dest_frame = ttk.LabelFrame(
            self.root, text=" 目标目录 ", padding=(10, 5)
        )
        dest_frame.place(x=10, y=170, width=680, height=70)

        # 目标目录输入框
        self.dest_entry = ttk.Entry(dest_frame, font=("Microsoft YaHei", 10))
        self.dest_entry.place(x=10, y=10, width=530, height=30)

        # 目标目录选择按钮
        dest_button = ttk.Button(
            dest_frame, text="选择目录", command=self.select_destination_directory
        )
        dest_button.place(x=550, y=10, width=100, height=30)

        # 选项框架
        options_frame = ttk.LabelFrame(
            self.root, text=" 选项 ", padding=(10, 5)
        )
        options_frame.place(x=10, y=250, width=680, height=60)

        # 移动文件复选框
        move_checkbox = ttk.Checkbutton(
            options_frame, text="移动文件（不选择则复制）", variable=self.move_files
        )
        move_checkbox.pack(anchor="w", padx=5, pady=5)

        # 操作按钮框架
        action_frame = ttk.Frame(self.root)
        action_frame.place(x=10, y=320, width=680, height=50)

        # 开始整理按钮
        start_button = ttk.Button(
            action_frame, text="开始整理", command=self.start_organizing
        )
        start_button.place(x=200, y=5, width=100, height=40)

        # 退出按钮
        exit_button = ttk.Button(
            action_frame, text="退出", command=self.root.quit
        )
        exit_button.place(x=380, y=5, width=100, height=40)

        # 进度条
        self.progress_bar = ttk.Progressbar(
            self.root, maximum=100, mode='determinate', style="Horizontal.TProgressbar"
        )
        self.progress_bar.place(x=10, y=380, width=680, height=30)

        # 日志显示框架
        log_frame = ttk.LabelFrame(
            self.root, text=" 日志 ", padding=(10, 5)
        )
        log_frame.place(x=10, y=420, width=680, height=70)

        # 日志文本框
        self.log_text = tk.Text(
            log_frame, height=4, font=("Microsoft YaHei", 9)
        )
        self.log_text.pack(fill="both", expand=True)

    # ------------------------------------------------------------------------------------------------------------------
    def add_source_directory(self):
        """
        添加源目录到列表
        """
        directory = filedialog.askdirectory()
        if directory:
            if directory not in self.source_directories:
                self.source_directories.append(directory)
                self.source_listbox.insert(tk.END, directory)
            else:
                messagebox.showinfo("提示", "该目录已添加。")

    # ------------------------------------------------------------------------------------------------------------------
    def remove_selected_source(self):
        """
        移除选中的源目录
        """
        selected_indices = self.source_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("警告", "请先选择要移除的目录。")
            return

        # 反向遍历，防止索引错位
        for index in reversed(selected_indices):
            self.source_listbox.delete(index)
            del self.source_directories[index]

    # ------------------------------------------------------------------------------------------------------------------
    def clear_source_directories(self):
        """
        清空源目录列表
        """
        confirm = messagebox.askyesno("确认", "确定要清空源目录列表吗？")
        if confirm:
            self.source_listbox.delete(0, tk.END)
            self.source_directories.clear()

    # ------------------------------------------------------------------------------------------------------------------
    def select_destination_directory(self):
        """
        选择目标目录
        """
        directory = filedialog.askdirectory()
        if directory:
            self.destination_directory = directory
            self.dest_entry.delete(0, tk.END)
            self.dest_entry.insert(0, directory)

    # ------------------------------------------------------------------------------------------------------------------
    def start_organizing(self):
        """
        开始整理文件
        """
        if not self.source_directories:
            messagebox.showwarning("警告", "请至少选择一个源目录。")
            return

        if not self.destination_directory:
            messagebox.showwarning("警告", "请选择目标目录。")
            return

        # 禁用控件，防止重复操作
        self.disable_controls()

        # 创建新线程，避免界面卡顿
        thread = threading.Thread(target=self.organize_files)
        thread.start()

    # ------------------------------------------------------------------------------------------------------------------
    def disable_controls(self):
        """
        禁用所有控件
        """
        for child in self.root.winfo_children():
            try:
                child_state = child.cget('state')
                if isinstance(child, (ttk.Frame, ttk.LabelFrame)):
                    for subchild in child.winfo_children():
                        subchild.configure(state='disabled')
                else:
                    child.configure(state='disabled')
            except tk.TclError:
                pass

    # ------------------------------------------------------------------------------------------------------------------
    def enable_controls(self):
        """
        启用所有控件
        """
        for child in self.root.winfo_children():
            try:
                if isinstance(child, (ttk.Frame, ttk.LabelFrame)):
                    for subchild in child.winfo_children():
                        subchild.configure(state='normal')
                else:
                    child.configure(state='normal')
            except tk.TclError:
                pass

    # ------------------------------------------------------------------------------------------------------------------
    def is_media_file(self, filename):
        """
        判断文件是否为媒体文件
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return True
        elif ext in VIDEO_EXTENSIONS:
            return True
        elif ext in AUDIO_EXTENSIONS:
            return True
        else:
            return False

    # ------------------------------------------------------------------------------------------------------------------
    def get_file_type(self, filename):
        """
        获取文件类型（图片、视频、音频）
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return '图片'
        elif ext in VIDEO_EXTENSIONS:
            return '视频'
        elif ext in AUDIO_EXTENSIONS:
            return '音频'
        else:
            return '其他'

    # ------------------------------------------------------------------------------------------------------------------
    def ensure_permissions(self, filepath):
        """
        确保有权限访问文件或目录
        """
        try:
            os.chmod(filepath, 0o755)
        except Exception as e:
            self.log(f"无法修改文件权限：{filepath}，错误：{e}")

    # ------------------------------------------------------------------------------------------------------------------
    def copy_or_move_file(self, src_path, dest_dir):
        """
        复制或移动文件到目标目录，处理重名文件
        """
        filename = os.path.basename(src_path)
        name, ext = os.path.splitext(filename)
        dest_path = os.path.join(dest_dir, filename)

        counter = 1
        # 如果目标文件已存在，添加“-数字”后缀
        while os.path.exists(dest_path):
            new_filename = f"{name}-{counter}{ext}"
            dest_path = os.path.join(dest_dir, new_filename)
            counter += 1

        try:
            if self.move_files.get():
                shutil.move(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)
        except Exception as e:
            self.log(f"处理文件失败：{src_path}，错误：{e}")

    # ------------------------------------------------------------------------------------------------------------------
    def log(self, message):
        """
        在日志窗口中显示信息
        """
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)

    # ------------------------------------------------------------------------------------------------------------------
    def organize_files(self):
        """
        整理文件的主函数，遍历源目录，处理媒体文件
        """
        total_files = 0       # 总文件数
        processed_files = 0   # 已处理文件数
        counts = defaultdict(int)  # 文件类型计数

        # 统计总的媒体文件数量
        for directory in self.source_directories:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if self.is_media_file(file):
                        total_files += 1

        if total_files == 0:
            self.log("未找到任何媒体文件。")
            self.enable_controls()
            return

        # 设置进度条
        self.progress_bar["maximum"] = total_files
        self.progress_bar["value"] = 0

        # 开始处理文件
        for directory in self.source_directories:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    # 检查主线程是否仍然活动
                    if not threading.main_thread().is_alive():
                        self.log("操作已取消。")
                        self.enable_controls()
                        return

                    src_path = os.path.join(root, file)
                    if self.is_media_file(file):
                        file_type = self.get_file_type(file)
                        counts[file_type] += 1

                        try:
                            mtime = os.path.getmtime(src_path)
                        except Exception as e:
                            self.log(f"获取修改时间失败：{src_path}，错误：{e}")
                            continue

                        # 根据修改时间创建月份文件夹
                        month_folder = datetime.fromtimestamp(
                            mtime).strftime('%Y-%m')

                        dest_dir = os.path.join(
                            self.destination_directory, month_folder)

                        # 确保目标目录存在
                        if not os.path.exists(dest_dir):
                            try:
                                os.makedirs(dest_dir)
                            except Exception as e:
                                self.log(f"创建目录失败：{dest_dir}，错误：{e}")
                                continue

                        # 确保有权限读取和写入
                        self.ensure_permissions(src_path)
                        self.ensure_permissions(dest_dir)

                        # 复制或移动文件并处理重名
                        self.copy_or_move_file(src_path, dest_dir)

                        processed_files += 1
                        self.update_progress(processed_files)
                        self.log(f"处理文件：{src_path}")

        # 处理完成
        self.log("文件整理完成！")
        self.log("统计结果：")
        for media_type, count in counts.items():
            self.log(f"{media_type}: {count} 个文件")

        messagebox.showinfo("完成", "文件整理完成！")
        self.enable_controls()

    # ------------------------------------------------------------------------------------------------------------------
    def update_progress(self, value):
        """
        更新进度条
        """
        self.progress_bar["value"] = value
        self.progress_bar.update_idletasks()
        self.root.update()

# ----------------------------------------------------------------------------------------------------------------------
def main():
    """
    主函数，启动应用程序
    """
    root = tk.Tk()
    app = MediaOrganizerGUI(root)
    root.mainloop()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
