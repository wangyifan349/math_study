# app.py
from flask import Flask, render_template_string, redirect, url_for, flash, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from urllib.parse import unquote

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # 请更改为您的密钥
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cloud_disk.db'
app.config['UPLOAD_FOLDER'] = 'uploads'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# 用户模型
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

# 创建数据库表
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# 允许的文件扩展名
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'zip', 'rar', 'mp4', 'mp3', 'avi', 'mkv'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------
# 路由和视图函数
# -------------------------------

# 注册
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('两次输入的密码不一致', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('用户名已存在', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        # 创建用户文件夹
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], username)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        flash('注册成功，请登录', 'success')
        return redirect(url_for('login'))

    # 注册页面模板
    register_html = '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>用户注册</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css">
    </head>
    <body>
        <div class="container mt-5" style="max-width:500px;">
            <h2 class="text-center">用户注册</h2>
            {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
            <div class="mt-3">
                {% for category, message in messages %}
                <div class="alert alert-{{ category }}" role="alert">
                    {{ message }}
                </div>
                {% endfor %}
            </div>
            {% endif %}
            {% endwith %}
            <form method="POST" action="">
                <div class="mb-3">
                    <label>用户名</label>
                    <input type="text" name="username" class="form-control" required>
                </div>
                <div class="mb-3">
                    <label>密码</label>
                    <input type="password" name="password" class="form-control" required>
                </div>
                <div class="mb-3">
                    <label>确认密码</label>
                    <input type="password" name="confirm_password" class="form-control" required>
                </div>
                <button type="submit" class="btn btn-primary w-100">注册</button>
            </form>
            <p class="mt-3 text-center">已有账户？<a href="{{ url_for('login') }}">登录</a></p>
        </div>
    </body>
    </html>
    '''
    return render_template_string(register_html)

# 登录
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误', 'danger')
            return redirect(url_for('login'))

    # 登录页面模板
    login_html = '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>用户登录</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css">
    </head>
    <body>
        <div class="container mt-5" style="max-width:500px;">
            <h2 class="text-center">用户登录</h2>
            {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
            <div class="mt-3">
                {% for category, message in messages %}
                <div class="alert alert-{{ category }}" role="alert">
                    {{ message }}
                </div>
                {% endfor %}
            </div>
            {% endif %}
            {% endwith %}
            <form method="POST" action="">
                <div class="mb-3">
                    <label>用户名</label>
                    <input type="text" name="username" class="form-control" required>
                </div>
                <div class="mb-3">
                    <label>密码</label>
                    <input type="password" name="password" class="form-control" required>
                </div>
                <button type="submit" class="btn btn-primary w-100">登录</button>
            </form>
            <p class="mt-3 text-center">还没有账户？<a href="{{ url_for('register') }}">注册</a></p>
        </div>
    </body>
    </html>
    '''
    return render_template_string(login_html)

# 登出
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# 主页面，文件列表
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
@login_required
def index(path):
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], current_user.username)
    current_path = os.path.join(user_folder, path)
    if not os.path.exists(current_path):
        flash('目录不存在', 'danger')
        return redirect(url_for('index'))

    # 获取目录下的文件和文件夹
    items = os.listdir(current_path)
    files = []
    folders = []
    for item in items:
        item_path = os.path.join(current_path, item)
        if os.path.isfile(item_path):
            files.append(item)
        elif os.path.isdir(item_path):
            folders.append(item)

    # 获取父路径
    parent_path = '/'.join(path.strip('/').split('/')[:-1])

    # 主页面模板
    index_html = '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>云盘</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css">
        <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js"></script>
        <style>
            body {
                user-select: none; /* 禁止文本选择，防止与右键菜单冲突 */
            }
            /* 自定义右键菜单样式 */
            #contextMenu, #blankContextMenu {
                display: none;
                position: absolute;
                z-index: 1000;
            }
            /* 防止表格中的文字被拖拽 */
            .table td {
                -webkit-user-drag: none;
            }
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <h2 class="mb-4">欢迎，{{ current_user.username }}！</h2>
            {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
            <div class="mt-3">
                {% for category, message in messages %}
                <div class="alert alert-{{ category }}" role="alert">
                    {{ message }}
                </div>
                {% endfor %}
            </div>
            {% endif %}
            {% endwith %}
            <div class="mb-3">
                <a href="{{ url_for('logout') }}" class="btn btn-secondary">退出登录</a>
                {% if path %}
                <a href="{{ url_for('index', path=parent_path) }}" class="btn btn-primary">返回上一级</a>
                {% else %}
                <span class="btn btn-secondary disabled">返回上一级</span>
                {% endif %}
            </div>
            <h5>当前位置：/{{ path }}</h5>
            <table class="table table-hover" id="fileTable">
                <thead>
                    <tr>
                        <th>名称</th>
                        <th>操作</th>
                    </tr>
                </thead>
                <tbody>
                    {% for folder in folders %}
                    <tr class="item-row" data-path="{{ path }}/{{ folder }}" data-type="folder">
                        <td>
                            <a href="{{ url_for('index', path=(path + '/' + folder)|strip('/')) }}"><i class="bi bi-folder-fill"></i> {{ folder }}</a>
                        </td>
                        <td></td>
                    </tr>
                    {% endfor %}
                    {% for file in files %}
                    <tr class="item-row" data-path="{{ path }}/{{ file }}" data-type="file">
                        <td>
                            {% if file.endswith(('.mp4', '.avi', '.mkv')) %}
                            <!-- 视频文件，提供在线播放 -->
                            <a href="#" onclick="playVideo('{{ url_for('download', filename=(path + '/' + file)) }}')"><i class="bi bi-film"></i> {{ file }}</a>
                            {% elif file.endswith(('.mp3', '.wav')) %}
                            <!-- 音频文件，提供在线播放 -->
                            <a href="#" onclick="playAudio('{{ url_for('download', filename=(path + '/' + file)) }}')"><i class="bi bi-music-note-beamed"></i> {{ file }}</a>
                            {% else %}
                            <i class="bi bi-file-earmark"></i> {{ file }}
                            {% endif %}
                        </td>
                        <td></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>

            <!-- 播放器模态框 -->
            <div class="modal fade" tabindex="-1" id="playerModal">
              <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content">
                  <div class="modal-body" id="playerContainer">
                  </div>
                  <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" onclick="closePlayer()">关闭</button>
                  </div>
                </div>
              </div>
            </div>
        </div>

        <!-- 自定义右键菜单 -->
        <div id="contextMenu" class="dropdown-menu">
            <a class="dropdown-item" href="#" id="renameItem">重命名</a>
            <a class="dropdown-item" href="#" id="deleteItem">删除</a>
        </div>

        <!-- 空白处右键菜单 -->
        <div id="blankContextMenu" class="dropdown-menu">
            <a class="dropdown-item" href="#" id="uploadItem">上传文件</a>
            <a class="dropdown-item" href="#" id="newFolderItem">新建文件夹</a>
        </div>

        <!-- JavaScript -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>
        <!-- 引入Bootstrap Icons -->
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.9.1/font/bootstrap-icons.css">
        <script>
            var currentItemPath = '';
            var currentItemType = '';

            $(document).ready(function() {
                // 阻止默认右键菜单
                $(document).on('contextmenu', function(e) {
                    e.preventDefault();
                });

                // 文件或文件夹右键菜单
                $('.item-row').on('contextmenu', function(e) {
                    currentItemPath = $(this).data('path');
                    currentItemType = $(this).data('type');
                    $('#contextMenu').css({
                        display: "block",
                        left: e.pageX,
                        top: e.pageY
                    });
                    $('#blankContextMenu').hide();
                });

                // 空白处右键菜单
                $('#fileTable').on('contextmenu', function(e) {
                    if ($(e.target).is('td')) {
                        currentItemPath = '';
                        $('#blankContextMenu').css({
                            display: "block",
                            left: e.pageX,
                            top: e.pageY
                        });
                        $('#contextMenu').hide();
                    }
                });

                // 点击空白处隐藏菜单
                $(document).click(function(e) {
                    if (!$(e.target).closest('.dropdown-menu').length) {
                        $('.dropdown-menu').hide();
                    }
                });

                // 删除
                $('#deleteItem').click(function() {
                    if (confirm('确认删除?')) {
                        $.ajax({
                            url: '/delete',
                            type: 'POST',
                            data: JSON.stringify({ 'path': currentItemPath, 'type': currentItemType }),
                            contentType: 'application/json;charset=UTF-8',
                            success: function(response) {
                                alert(response.message);
                                location.reload();
                            }
                        });
                    }
                });

                // 重命名
                $('#renameItem').click(function() {
                    var newName = prompt('输入新名称：');
                    if (newName) {
                        $.ajax({
                            url: '/rename',
                            type: 'POST',
                            data: JSON.stringify({ 'path': currentItemPath, 'new_name': newName, 'type': currentItemType }),
                            contentType: 'application/json;charset=UTF-8',
                            success: function(response) {
                                alert(response.message);
                                location.reload();
                            }
                        });
                    }
                });

                // 上传文件
                $('#uploadItem').click(function() {
                    $('#uploadModal').modal('show');
                });

                // 新建文件夹
                $('#newFolderItem').click(function() {
                    createFolderPrompt();
                });
            });

            function createFolderPrompt() {
                var folderName = prompt('输入新文件夹的名称：');
                if (folderName) {
                    $.ajax({
                        url: '/create_folder',
                        type: 'POST',
                        data: JSON.stringify({ 'path': "{{ path }}", 'folder_name': folderName }),
                        contentType: 'application/json;charset=UTF-8',
                        success: function(response) {
                            alert(response.message);
                            location.reload();
                        }
                    });
                }
            }

            function playVideo(url) {
                var videoHtml = '<video width="100%" controls autoplay><source src="' + url + '" type="video/mp4">您的浏览器不支持video标签。</video>';
                $('#playerContainer').html(videoHtml);
                var playerModal = new bootstrap.Modal(document.getElementById('playerModal'));
                playerModal.show();
            }

            function playAudio(url) {
                var audioHtml = '<audio width="100%" controls autoplay><source src="' + url + '" type="audio/mpeg">您的浏览器不支持audio标签。</audio>';
                $('#playerContainer').html(audioHtml);
                var playerModal = new bootstrap.Modal(document.getElementById('playerModal'));
                playerModal.show();
            }

            function closePlayer() {
                var playerModal = bootstrap.Modal.getInstance(document.getElementById('playerModal'));
                playerModal.hide();
                $('#playerContainer').html('');
            }
        </script>

        <!-- 上传文件模态框 -->
        <div class="modal fade" id="uploadModal" tabindex="-1">
          <div class="modal-dialog">
            <div class="modal-content">
              <form method="POST" enctype="multipart/form-data" action="{{ url_for('upload', path=path) }}">
              <div class="modal-header">
                <h5 class="modal-title">上传文件</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="关闭"></button>
              </div>
              <div class="modal-body">
                <div class="mb-3">
                    <input type="file" name="file" class="form-control">
                </div>
              </div>
              <div class="modal-footer">
                <button type="submit" class="btn btn-primary">上传</button>
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
              </div>
              </form>
            </div>
          </div>
        </div>

    </body>
    </html>
    '''

    return render_template_string(index_html, path=path, files=files, folders=folders, parent_path=parent_path)

# 上传文件
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    path = request.args.get('path', '')
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], current_user.username)
    current_path = os.path.join(user_folder, path)

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('没有选择文件', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('没有选择文件', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(current_path, filename))
            flash('文件上传成功', 'success')
            return redirect(url_for('index', path=path))
        else:
            flash('文件类型不允许', 'danger')
            return redirect(request.url)

    # 上传页面模板
    upload_html = '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>上传文件</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css">
    </head>
    <body>
        <div class="container mt-5" style="max-width:500px;">
            <h2>上传文件</h2>
            {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
            <div class="mt-3">
                {% for category, message in messages %}
                <div class="alert alert-{{ category }}" role="alert">
                    {{ message }}
                </div>
            {% endfor %}
            </div>
            {% endif %}
            {% endwith %}
            <form method="POST" enctype="multipart/form-data">
                <div class="mb-3">
                    <input type="file" name="file" class="form-control">
                </div>
                <button type="submit" class="btn btn-primary w-100">上传</button>
            </form>
        </div>
    </body>
    </html>
    '''

    return render_template_string(upload_html)

# 下载文件
@app.route('/download/<path:filename>')
@login_required
def download(filename):
    filename = unquote(filename)
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], current_user.username)
    file_path = os.path.join(user_folder, filename)

    if os.path.exists(file_path):
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        return send_from_directory(directory, filename, as_attachment=False)
    else:
        flash('文件不存在', 'danger')
        return redirect(url_for('index'))

# 删除文件或文件夹
@app.route('/delete', methods=['POST'])
@login_required
def delete():
    data = request.get_json()
    path = data['path']
    item_type = data['type']  # 'file' or 'folder'

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], current_user.username)
    item_path = os.path.join(user_folder, path.strip('/'))

    if os.path.exists(item_path):
        try:
            if item_type == 'file':
                os.remove(item_path)
            elif item_type == 'folder':
                if not os.listdir(item_path):
                    os.rmdir(item_path)
                else:
                    import shutil
                    shutil.rmtree(item_path)
            return jsonify({'message': '删除成功'})
        except Exception as e:
            return jsonify({'message': '删除失败：' + str(e)})
    else:
        return jsonify({'message': '路径不存在'})

# 重命名文件或文件夹
@app.route('/rename', methods=['POST'])
@login_required
def rename():
    data = request.get_json()
    path = data['path']
    new_name = data['new_name']
    item_type = data['type']

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], current_user.username)
    old_path = os.path.join(user_folder, path.strip('/'))
    new_path = os.path.join(os.path.dirname(old_path), secure_filename(new_name))

    if os.path.exists(old_path):
        try:
            os.rename(old_path, new_path)
            return jsonify({'message': '重命名成功'})
        except Exception as e:
            return jsonify({'message': '重命名失败：' + str(e)})
    else:
        return jsonify({'message': '路径不存在'})

# 创建文件夹
@app.route('/create_folder', methods=['POST'])
@login_required
def create_folder():
    data = request.get_json()
    path = data['path']
    folder_name = data['folder_name']

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], current_user.username)
    folder_path = os.path.join(user_folder, path.strip('/'), secure_filename(folder_name))

    if not os.path.exists(folder_path):
        try:
            os.mkdir(folder_path)
            return jsonify({'message': '文件夹创建成功'})
        except Exception as e:
            return jsonify({'message': '文件夹创建失败：' + str(e)})
    else:
        return jsonify({'message': '文件夹已存在'})

# 运行应用
if __name__ == '__main__':
    app.run(debug=True)
