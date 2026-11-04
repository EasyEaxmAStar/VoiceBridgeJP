"""Camera controls for the existing MainWindow. Only bounded metadata on Qt."""
import copy,time
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QGroupBox,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QCheckBox,QDialog,
    QComboBox,QFormLayout,QWidget,QScrollArea,QSpinBox,QDoubleSpinBox)
from .camera_settings import defaults,validate,PRESETS


class CameraDialog(QDialog):
    def __init__(self,client,state=None,parent=None):
        super().__init__(parent);self.client=client;self.state=state or {};self.loading=False
        self.setWindowTitle('VoiceBridge JP · 摄像头设置');available=self.screen().availableGeometry()
        self.resize(min(900,available.width()-32),min(790,available.height()-64))
        from .gui_theme import settings_theme
        settings_theme(self)
        outer=QVBoxLayout(self);self.notice=QLabel();self.notice.setWordWrap(True);outer.addWidget(self.notice)
        scroll=QScrollArea();scroll.setWidgetResizable(True);body=QWidget();form=QFormLayout(body);scroll.setWidget(body);outer.addWidget(scroll)
        self.enabled=QCheckBox('启用 Smart Camera Glitch（独立视频进程）');form.addRow(self.enabled)
        self.physical=QComboBox();self.physical.addItem('请选择物理摄像头','')
        self.physical.setMinimumContentsLength(22);self.physical.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.refresh=QPushButton('刷新摄像头');self.refresh.clicked.connect(self.enumerate)
        row=QHBoxLayout();row.addWidget(self.physical,1);row.addWidget(self.refresh);form.addRow('物理摄像头',row)
        self.virtual=QComboBox();self.virtual.addItem('OBS Virtual Camera',['obs','OBS Virtual Camera']);form.addRow('虚拟输出设备',self.virtual)
        self.backend=QComboBox();self.backend.addItem('CPU BGR (Stable — 推荐)','bgr');self.backend.addItem('NV12 (Experimental — colors may differ)','nv12');form.addRow('处理路径',self.backend)
        self.warning=QLabel('NV12 是实验性CPU像素格式转换，可能减少转换开销，也可能产生色差；不是Intel GPU加速。建议CPU BGR。');self.warning.setWordWrap(True);form.addRow(self.warning)
        self.confirm=QCheckBox('我了解实验色差，同意在本次启动试用 NV12');form.addRow(self.confirm)
        self.backend.currentIndexChanged.connect(self.show_warning)
        self.resolution=QComboBox()
        for label,size in [('640 × 480',[640,480]),('1280 × 720（推荐）',[1280,720]),('1920 × 1080（需设备验证）',[1920,1080])]:self.resolution.addItem(label,size)
        form.addRow('分辨率',self.resolution)
        self.probe=QPushButton('验证此设备1080p（会重启视频，音频模型保留）')
        self.probe.clicked.connect(lambda:self.client.send('camera_probe1080',identity=self.physical.currentData() or ''));form.addRow(self.probe)
        self.fps=QSpinBox();self.fps.setRange(15,60);self.fps.setSuffix(' fps（目标虚拟输出）');form.addRow('目标输出帧率',self.fps)
        self.intensity=QComboBox()
        for label,value in [('低 / Low','LOW'),('中 / Medium','MEDIUM'),('高 / High','HIGH'),('自定义 / Custom','CUSTOM')]:self.intensity.addItem(label,value)
        form.addRow('效果强度',self.intensity);self.intensity.currentIndexChanged.connect(self.preset)
        self.continuous=QCheckBox('识别期间持续可见：全画面基础马赛克（推荐用于明显联动）')
        self.continuous.setToolTip('按所选触发时机进入效果，第一帧即像素化；实际非豁免播放开始即恢复当前清晰画面。')
        form.addRow(self.continuous)
        self.strong=QCheckBox('强烈卡顿：整帧冻结后跳帧（建议冻结500–800ms）')
        self.strong.setToolTip('效果激活后立即冻结整帧；按下方冻结范围跳到新画面。仍以目标FPS连续发送重复帧；CLEAN立即解除冻结。')
        form.addRow(self.strong)
        self.trigger=QComboBox();self.trigger.addItem('麦克风声音起始（推荐）','microphone');self.trigger.addItem('ASR识别开始（旧版）','asr');form.addRow('触发时机',self.trigger)
        self.threshold=QDoubleSpinBox();self.threshold.setDecimals(3);self.threshold.setRange(.001,.1);self.threshold.setSingleStep(.001);form.addRow('声音阈值 RMS（越低越敏感）',self.threshold)
        self.region_min=QSpinBox();self.region_max=QSpinBox()
        for widget in (self.region_min,self.region_max):widget.setRange(1,12)
        row=QHBoxLayout();row.addWidget(self.region_min);row.addWidget(QLabel('至'));row.addWidget(self.region_max);form.addRow('马赛克区域数',row)
        self.block=QSpinBox();self.block.setRange(1,64);self.block.setSuffix(' px（最大块边长）');form.addRow('马赛克强度',self.block)
        self.stutter=QDoubleSpinBox();self.stutter.setRange(0,1);self.stutter.setSingleStep(.1);form.addRow('随机卡顿强度（0–1）',self.stutter)
        self.freeze_min=QSpinBox();self.freeze_max=QSpinBox()
        for widget in (self.freeze_min,self.freeze_max):widget.setRange(50,800);widget.setSuffix(' ms')
        row=QHBoxLayout();row.addWidget(self.freeze_min);row.addWidget(QLabel('至'));row.addWidget(self.freeze_max);form.addRow('冻结时长',row)
        self.randomness=QDoubleSpinBox();self.randomness.setRange(0,1);self.randomness.setSingleStep(.1);form.addRow('时序随机变化幅度（0–1）',self.randomness)
        self.timeout=QSpinBox();self.timeout.setRange(1,120);self.timeout.setSuffix(' s');form.addRow('故障效果硬超时',self.timeout)
        self.report=QLabel();self.report.setWordWrap(True);outer.addWidget(self.report)
        buttons=QHBoxLayout();self.apply=QPushButton('应用并保存');self.apply.clicked.connect(self.save)
        reset=QPushButton('恢复摄像头默认值');reset.clicked.connect(lambda:self.load(dict(camera=dict(settings=defaults()))))
        help_button=QPushButton('本地预览说明');help_button.clicked.connect(self.preview_help)
        close=QPushButton('关闭');close.clicked.connect(self.close)
        for widget in (reset,help_button,self.apply,close):buttons.addWidget(widget)
        outer.addLayout(buttons)
        for widget in (self.region_min,self.region_max,self.block,self.stutter,self.freeze_min,self.freeze_max,self.randomness):
            widget.valueChanged.connect(self.custom)
        self.load(dict(camera=self.state or dict(settings=defaults())))

    def show_warning(self):
        enabled=self.backend.currentData()=='nv12';self.warning.setVisible(enabled);self.confirm.setVisible(enabled)
    def custom(self):
        if not self.loading:self.intensity.setCurrentIndex(self.intensity.findData('CUSTOM'))
    def preset(self):
        if self.loading:return
        level=self.intensity.currentData()
        if level in PRESETS:
            self.loading=True;self._effects(PRESETS[level]);self.loading=False
    def _effects(self,v):
        if 'continuous_mosaic' in v:self.continuous.setChecked(v['continuous_mosaic'])
        if 'strong_stutter' in v:self.strong.setChecked(v['strong_stutter'])
        self._block_min=v['block_min']
        self.region_min.setValue(v['regions_min']);self.region_max.setValue(v['regions_max']);self.block.setValue(v['block_max'])
        self.stutter.setValue(v['stutter_intensity']);self.freeze_min.setValue(round(v['freeze_min']*1000));self.freeze_max.setValue(round(v['freeze_max']*1000));self.randomness.setValue(v['randomness'])

    def load(self,message):
        state=message.get('camera',{});value=state.get('settings',defaults());self.state=copy.deepcopy(state);self.value=copy.deepcopy(value)
        self.loading=True;self.enabled.setChecked(value['enabled']);v=value['video']
        index=self.physical.findData(v['camera_identity'])
        if index<0:
            self.physical.addItem('[已保存设备 / 待枚举]',v['camera_identity']);index=self.physical.count()-1
            self.physical.setItemData(index,v['camera_identity'],Qt.ToolTipRole)
        self.physical.setCurrentIndex(index)
        virtual=[v['virtual_backend'],v['virtual_device']];index=self.virtual.findData(virtual)
        if index<0:self.virtual.addItem('[已保存] '+v['virtual_device'],virtual);index=self.virtual.count()-1
        self.virtual.setCurrentIndex(index);self.backend.setCurrentIndex(self.backend.findData(v['processing_backend']))
        self.resolution.setCurrentIndex(self.resolution.findData([v['width'],v['height']]))
        self.resolution.model().item(2).setEnabled(bool(state.get('verified_1080')))
        self.resolution.setToolTip('1080p仅在该设备实际验证后可选；不会静默缩放或降分辨率。')
        self.fps.setValue(v['fps']);self.intensity.setCurrentIndex(self.intensity.findData(value['intensity']));self._effects(v);self.timeout.setValue(round(value['safety_timeout']))
        self.trigger.setCurrentIndex(self.trigger.findData(v.get('trigger_mode','microphone')));self.threshold.setValue(v.get('onset_threshold',.012))
        self.confirm.setChecked(False);self.loading=False;self.show_warning()
        locked=state.get('environment_locked',False)
        self.apply.setEnabled(not locked);self.notice.setText('设置来源：'+state.get('source','安全默认值')+(' · 启动环境覆盖生效；清除环境变量并重启后可编辑。可在主界面停止摄像头。' if locked else ' · 仅重启视频进程，保留音频及所有模型。'))
        self.report.setText(state.get('notice') or state.get('error') or '修改草稿不会打开摄像头；应用后才生效。')

    def enumerate(self):
        self.report.setText('正在后台枚举，不会打开摄像头…');self.refresh.setEnabled(False)
        if not self.client.send('camera_devices'):self.refresh.setEnabled(True);self.report.setText('控制队列繁忙，请稍后重试。')
    def devices(self,data):
        current=self.physical.currentData();self.physical.clear();self.physical.addItem('请选择物理摄像头','')
        for device in data.get('cameras',[]):
            if not device['virtual'] and not device.get('infrared',False):self.physical.addItem(device['name'],device['identity'])
        index=self.physical.findData(current)
        if index<0:
            self.physical.addItem('[已保存设备 / 未连接]',current);index=self.physical.count()-1;self.physical.setItemData(index,str(current),Qt.ToolTipRole)
        self.physical.setCurrentIndex(index);self.refresh.setEnabled(True)
        obs=any(b.get('backend')=='obs' for b in data.get('virtual_backends',[]))
        self.report.setText('OBS虚拟后端已检出；实际占用情况在启用时检查。' if obs else '未检出OBS Virtual Camera；请安装/修复官方OBS驱动后重试，本程序不会自动安装。')
    def draft(self):
        value=copy.deepcopy(self.value);v=value['video'];width,height=self.resolution.currentData();backend,device=self.virtual.currentData()
        value.update(enabled=self.enabled.isChecked(),intensity=self.intensity.currentData(),safety_timeout=self.timeout.value())
        v.update(camera_identity=self.physical.currentData() or '',width=width,height=height,fps=self.fps.value(),virtual_backend=backend,virtual_device=device,
            processing_backend=self.backend.currentData(),regions_min=self.region_min.value(),regions_max=self.region_max.value(),
            block_min=min(self._block_min,self.block.value()),block_max=self.block.value(),stutter_intensity=self.stutter.value(),
            continuous_mosaic=self.continuous.isChecked(),
            strong_stutter=self.strong.isChecked(),
            trigger_mode=self.trigger.currentData(),onset_threshold=self.threshold.value(),
            freeze_min=self.freeze_min.value()/1000,freeze_max=self.freeze_max.value()/1000,randomness=self.randomness.value())
        return validate(value)
    def save(self):
        try:
            value=self.draft()
            if value['video']['processing_backend']=='nv12' and not self.confirm.isChecked() and self.state.get('nv12_confirmation_required',True):
                raise ValueError('请明确勾选本次启动试用NV12；否则使用CPU BGR。')
            if not self.client.send('camera_save',settings=value,confirm_nv12=self.confirm.isChecked()):raise ValueError('控制队列繁忙')
            self.report.setText('正在安全应用摄像头配置…')
        except (ValueError,TypeError) as exc:self.report.setText('未应用：'+str(exc))
    def preview_help(self):
        self.report.setText('本地预览：在Zoom/Teams的视频设置选择OBS Virtual Camera，保持本地预览，不要加入会议或再次选择Integrated Camera。此界面仅传状态，不记录或上传视频；同一物理摄像头只有一个拥有者。')


class CameraPanel(QGroupBox):
    def __init__(self,client,parent=None):
        super().__init__('摄像头 · Smart Camera Glitch',parent);self.client=client;self.snapshot={};self.dialog=None;self.last_update=0;self.ready=False;self.pending_enable=False
        layout=QVBoxLayout(self);row=QHBoxLayout();self.enabled=QCheckBox('启用摄像头');row.addWidget(self.enabled)
        self.settings=QPushButton('摄像头设置');self.settings.clicked.connect(self.open_settings);row.addWidget(self.settings)
        self.retry=QPushButton('重试摄像头');self.retry.clicked.connect(lambda:self.client.send('camera_retry'));row.addWidget(self.retry)
        self.stop=QPushButton('停止摄像头');self.stop.clicked.connect(lambda:self.client.send('camera_disable'));row.addWidget(self.stop);layout.addLayout(row)
        self.summary=QLabel('DISABLED · CPU BGR稳定模式 · 物理/虚拟帧率待测');self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.notice=QLabel('在摄像头设置中明确选择设备；默认关闭。');self.notice.setWordWrap(True);layout.addWidget(self.notice)
        self.enabled.toggled.connect(self.toggle);self.set_ready(False)
    def set_ready(self,ready):
        self.ready=ready;locked=self.snapshot.get('environment_locked',False)
        self.enabled.setEnabled(ready and not locked);self.retry.setEnabled(ready);self.stop.setEnabled(ready)
        if self.dialog:
            self.dialog.apply.setEnabled(ready and not locked);self.dialog.refresh.setEnabled(ready);self.dialog.probe.setEnabled(ready and not locked)
    def toggle(self,enabled):
        if not self.snapshot:
            self.notice.setText('请先打开摄像头设置，明确选择物理摄像头。');self.enabled.blockSignals(True);self.enabled.setChecked(False);self.enabled.blockSignals(False);return
        value=copy.deepcopy(self.snapshot['settings']);value['enabled']=enabled
        if enabled and not value['video']['camera_identity']:
            self.enabled.blockSignals(True);self.enabled.setChecked(False);self.enabled.blockSignals(False)
            self.pending_enable=True;self.open_settings();self.dialog.enabled.setChecked(True);return
        self.client.send('camera_save',settings=value)
    def open_settings(self):
        if self.dialog is None:self.dialog=CameraDialog(self.client,self.snapshot,parent=self.window())
        else:self.dialog.load(dict(camera=self.snapshot or dict(settings=defaults())))
        self.set_ready(self.ready)
        self.dialog.show();self.dialog.raise_();self.client.send('camera_get')
    def update_state(self,state,*,force=False):
        if not state:return
        self.snapshot=copy.deepcopy(state)
        if not force and time.monotonic()-self.last_update<.5:return
        self.last_update=time.monotonic();stats=state.get('stats',{});output=stats.get('output',{})
        self.enabled.blockSignals(True);self.enabled.setChecked(state.get('enabled',False));self.enabled.blockSignals(False);self.set_ready(self.ready)
        def number(value):return '—' if value is None else f'{value:.1f}'
        mode=state.get('controller_mode',state.get('mode','DISABLED'));mode='DISABLED' if mode=='OFF' else mode
        self.summary.setText(f"控制 {mode} · 视频回报 {stats.get('mode','待就绪')}（最多1秒延迟） · 请求 {state.get('requested_backend','bgr').upper()} / 实际 {(state.get('active_backend') or '待就绪').upper()}\n物理交付 {number(stats.get('capture_observed_fps'))} fps · 内容变化 {number(stats.get('content_change_fps'))} fps · 虚拟发送 {number(stats.get('observed_fps'))} fps（可重复） · 帧龄P95 {number(stats.get('p95_frame_age_ms'))} ms\n采集 {'READY' if state.get('capture_ready') else 'NOT READY'} · 虚拟输出 {state.get('virtual_state','INITIALIZING')} · 额外周期迟到 {stats.get('deadline_misses',0)} / 发送失败 {stats.get('output_misses',0)}")
        continuous=state.get('settings',{}).get('video',{}).get('continuous_mosaic',False)
        strong=state.get('settings',{}).get('video',{}).get('strong_stutter',False)
        trigger=state.get('settings',{}).get('video',{}).get('trigger_mode','microphone')
        self.notice.setText(state.get('error') or state.get('notice') or ('来源：'+state.get('source','GUI')+f" · 强烈卡顿{'开' if strong else '关'} / 持续马赛克{'开' if continuous else '关'} · {'麦克风声音起始' if trigger=='microphone' else '旧版ASR开始'}触发；quick.hai豁免。"))
    def message(self,message):
        kind=message.get('type')
        if kind=='camera_settings':
            self.update_state(message['camera'],force=True)
            if self.dialog:
                self.dialog.load(message)
                if self.pending_enable:self.dialog.enabled.setChecked(True);self.pending_enable=False
        elif kind=='camera_devices' and self.dialog:self.dialog.devices(message['devices'])
        elif kind in {'camera_error','error'} and self.dialog:
            self.dialog.refresh.setEnabled(True);self.dialog.report.setText('仅本地：'+message.get('message','摄像头错误'))
