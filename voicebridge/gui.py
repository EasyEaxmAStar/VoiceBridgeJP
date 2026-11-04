"""Native Windows desktop shell for the tested hot audio routing controller."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
from PySide6.QtCore import Qt,QTimer,Signal
from PySide6.QtGui import QShortcut,QKeySequence
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,
    QLabel,QPushButton,QComboBox,QProgressBar,QGroupBox,QPlainTextEdit,QDoubleSpinBox,QCheckBox,QScrollArea,QFrame)
from .gui_client import ControllerClient
from .release_voices import DEFAULT,NAMES

LABELS={'OFFICIAL_V2PROPLUS':'GPT-SoVITS 官方 v2ProPlus'}
_APPLICATION=None

class BoundaryTrigger(QPushButton):
    pressed_boundary=Signal(str,float)
    def mousePressEvent(self,event):
        if event.button() in {Qt.LeftButton,Qt.RightButton}:
            self.pressed_boundary.emit('LEFT' if event.button()==Qt.LeftButton else 'RIGHT',time.perf_counter());event.accept()
        else:super().mousePressEvent(event)


class MainWindow(QMainWindow):
    status_received=Signal(dict)
    global_button=Signal(str,float,int)

    def __init__(self,client=None,*,preset=DEFAULT,connect=None,port=8765,fixture_input=False):
        super().__init__()
        # Keep the native application alive independently of test/local frames.
        global _APPLICATION
        _APPLICATION=QApplication.instance()
        self.setAttribute(Qt.WA_DeleteOnClose,True)
        self.client=client or ControllerClient(preset=preset,connect=connect,port=port,fixture_input=fixture_input)
        self.status={};self.selected_route='TRANSLATE';self.close_allowed=False;self.closing=False;self.requested_devices=False
        self.setWindowTitle('VoiceBridge JP · GPT-SoVITS')
        available=self.screen().availableGeometry();self.resize(min(1020,available.width()-32),min(850,available.height()-64))
        root=QWidget();root.setObjectName('mainSurface');self.main_scroll=QScrollArea();self.main_scroll.setObjectName('mainScroll')
        self.main_scroll.setWidgetResizable(True);self.main_scroll.setFrameShape(QFrame.NoFrame);self.main_scroll.setWidget(root);self.setCentralWidget(self.main_scroll)
        layout=QVBoxLayout(root);layout.setContentsMargins(24,18,24,18);layout.setSpacing(12)
        header=QHBoxLayout();title=QLabel('VoiceBridge JP');title.setObjectName('title');header.addWidget(title);header.addStretch()
        self.state=QLabel('AI INITIALIZING');self.state.setObjectName('state');header.addWidget(self.state);layout.addLayout(header)
        self.banner=QLabel('正在载入并预热本地模型 · 尚未发送麦克风音频');self.banner.setWordWrap(True);layout.addWidget(self.banner)
        routes=QHBoxLayout();self.translate=QPushButton('TRANSLATE\n中文 → 日语合成语音');self.direct=QPushButton('DIRECT\n原始麦克风 → VB-CABLE')
        for button in (self.translate,self.direct):button.setCheckable(True);button.setMinimumHeight(65);routes.addWidget(button)
        self.translate.setChecked(True);self.translate.clicked.connect(lambda:self.choose_route('TRANSLATE'))
        self.direct.clicked.connect(lambda:self.choose_route('DIRECT'));layout.addLayout(routes)
        controls=QHBoxLayout();self.buttons={}
        for name,label in [('start','Start 开始'),('pause','Pause 暂停'),('resume','Resume 继续'),('stop','Stop 停止'),('test','静音设备检查'),('recover','恢复')]:
            button=QPushButton(label);self.buttons[name]=button;controls.addWidget(button)
            button.clicked.connect(lambda checked=False,n=name:self.action(n))
        layout.addLayout(controls)
        quickbox=QGroupBox('快捷日语回复 / Quick Japanese Replies');quickrow=QHBoxLayout(quickbox)
        self.quick_buttons=[]
        from .quick_reply import SHORTCUTS
        for shortcut_id,phrase in SHORTCUTS.items():
            button=QPushButton('快速回复：'+phrase);button.clicked.connect(lambda checked=False,text=phrase,sid=shortcut_id:self.submit_quick(text,shortcut_id=sid))
            quickrow.addWidget(button);self.quick_buttons.append(button)
        self.quick_status=QLabel('发送合成日语到 VB-CABLE；仅活动 TRANSLATE 可用');self.quick_status.setWordWrap(True)
        quickrow.addWidget(self.quick_status);layout.addWidget(quickbox)
        backendrow=QHBoxLayout();self.translation_backend=QComboBox();self.translation_backend.addItem('OPUS Fast','OPUS_FAST');self.translation_backend.addItem('Qwen Fallback（审计警告不阻断）','QWEN_FALLBACK')
        self.backend_apply=QPushButton('暂停后应用翻译后端');self.backend_apply.clicked.connect(lambda:self.client.send('translation_backend',mode=self.translation_backend.currentData()))
        backendrow.addWidget(self.translation_backend);backendrow.addWidget(self.backend_apply);layout.addLayout(backendrow)
        devicebox=QGroupBox('音频设备 · Windows WASAPI');grid=QGridLayout(devicebox)
        self.microphones=QComboBox();self.microphones.setMinimumContentsLength(35)
        self.apply_mic=QPushButton('应用麦克风');self.refresh=QPushButton('刷新设备')
        grid.addWidget(QLabel('物理麦克风'),0,0);grid.addWidget(self.microphones,0,1);grid.addWidget(self.apply_mic,0,2);grid.addWidget(self.refresh,0,3)
        self.output=QLabel('固定输出：CABLE Input（正在验证）');grid.addWidget(self.output,1,0,1,4)
        self.input_meter=QProgressBar();self.output_meter=QProgressBar()
        for m in (self.input_meter,self.output_meter):m.setRange(0,100);m.setTextVisible(False);m.setMaximumHeight(10)
        grid.addWidget(QLabel('麦克风电平'),2,0);grid.addWidget(self.input_meter,2,1,1,3)
        grid.addWidget(QLabel('输出电平'),3,0);grid.addWidget(self.output_meter,3,1,1,3);layout.addWidget(devicebox)
        voicebox=QGroupBox('官方预训练语音');row=QHBoxLayout(voicebox);self.voices=QComboBox()
        for name in NAMES:self.voices.addItem(LABELS[name],name)
        self.voices.setCurrentIndex(self.voices.findData(preset));self.apply_voice=QPushButton('应用音色')
        row.addWidget(self.voices);row.addWidget(self.apply_voice);row.addSpacing(12);row.addWidget(QLabel('保音高播放速度'))
        self.speed=QDoubleSpinBox();self.speed.setRange(.75,1.5);self.speed.setSingleStep(.05);self.speed.setValue(1.25);self.speed.setSuffix(' ×');row.addWidget(self.speed)
        self.adaptive=QCheckBox('实验性自适应');row.addWidget(self.adaptive);layout.addWidget(voicebox)
        from .gui_camera import CameraPanel
        self.camera_panel=CameraPanel(self.client,self);layout.addWidget(self.camera_panel)
        self.settings_button=QPushButton('声音变化 / 词典 / 分段 / 日语语体设置');layout.addWidget(self.settings_button)
        self.settings_dialog=None;self.settings_button.clicked.connect(self.open_settings)
        self.mouse_hook=None;self._hook_generation=None;self._hook_failure=None
        boundary_box=QGroupBox('话语边界 · AUTO / MANUAL / HYBRID');boundary_layout=QHBoxLayout(boundary_box)
        self.boundary_info=QLabel('AUTO · DISARMED');self.boundary_info.setWordWrap(True);boundary_layout.addWidget(self.boundary_info,1)
        self.arm_button=QPushButton('ARM 启用点击');self.arm_button.clicked.connect(self.toggle_boundary_arm);boundary_layout.addWidget(self.arm_button)
        self.trigger=BoundaryTrigger('鼠标触发区域\n按设置的左 / 右键结束当前句');self.trigger.setMinimumHeight(50);self.trigger.pressed_boundary.connect(self.in_app_boundary);boundary_layout.addWidget(self.trigger)
        self.end_button=QPushButton('结束当前句\nCtrl+Enter');self.end_button.setMinimumHeight(50);self.end_button.clicked.connect(lambda:self.commit_boundary('manual_button'));boundary_layout.addWidget(self.end_button)
        self.end_shortcut=QShortcut(QKeySequence('Ctrl+Return'),self);self.end_shortcut.activated.connect(lambda:self.commit_boundary('keyboard'))
        self.global_button.connect(self.on_global_button)
        layout.addWidget(boundary_box)
        self.models=QLabel('ASR / MT / TTS：等待真实预热');self.gpu=QLabel('GPU 显存：等待采样');self.metrics=QLabel('队列：0 · 最新延迟：—')
        for label in (self.models,self.gpu,self.metrics):label.setWordWrap(True);layout.addWidget(label)
        previews=QHBoxLayout();self.source=QPlainTextEdit();self.japanese=QPlainTextEdit()
        for widget,label in ((self.source,'原始中文 ASR（仅最终提交文本）'),(self.japanese,'日语译文')):
            box=QGroupBox(label);v=QVBoxLayout(box);widget.setReadOnly(True);widget.setMaximumBlockCount(100);v.addWidget(widget);previews.addWidget(box)
        layout.addLayout(previews,1)
        self.logs=QPlainTextEdit();self.logs.setReadOnly(True);self.logs.setMaximumBlockCount(400);self.logs.setMaximumHeight(105);layout.addWidget(self.logs)
        self.clear_diagnostics=QPushButton('清空本地显示诊断（不产生声音）');self.clear_diagnostics.clicked.connect(self.logs.clear);layout.addWidget(self.clear_diagnostics)
        self.paths=QLabel(f'本地日志：{self.client.directory}');self.paths.setTextInteractionFlags(Qt.TextSelectableByMouse);self.paths.setWordWrap(True);layout.addWidget(self.paths)
        self.setStyleSheet('QWidget{color:#24364b} QMainWindow,QWidget#mainSurface,QScrollArea#mainScroll{background:#f3f5f8} QComboBox,QDoubleSpinBox{background:white;padding:5px;border:1px solid #bbc7d7;border-radius:5px} '
            'QGroupBox{font-weight:600;border:1px solid #ccd4df;border-radius:8px;margin-top:10px;padding-top:14px} '
            'QPushButton{padding:9px;border-radius:6px;border:1px solid #bbc7d7;background:white} QPushButton:checked{background:#174b79;color:white} '
            'QPushButton:disabled{color:#89929e;background:#e7ebf1} QLabel#title{font-size:26px;font-weight:700;color:#173d60} QLabel#state{font-weight:700} '
            'QPlainTextEdit{background:white;border:1px solid #ccd4df;border-radius:5px;padding:5px} QProgressBar::chunk{background:#2d8b77}')
        self.refresh.clicked.connect(lambda:self.client.send('devices'))
        self.apply_mic.clicked.connect(self.change_microphone);self.apply_voice.clicked.connect(self.change_voice)
        self.speed.valueChanged.connect(lambda x:self.client.send('speed',value=x));self.adaptive.toggled.connect(lambda b:self.client.send('adaptive',enabled=b))
        self.client.message.connect(self.on_message);self.client.failed.connect(self.on_error);self.client.exited.connect(self.on_exit)
        self.set_enabled(False);QTimer.singleShot(0,self.client.start)

    def set_enabled(self,ready):
        self.camera_panel.set_ready(ready and not self.closing)
        for button in self.quick_buttons:button.setEnabled(False)
        for widget in [self.translate,self.direct,self.refresh,self.apply_mic,self.apply_voice,self.voices,self.microphones,self.speed,self.adaptive,self.arm_button,self.end_button,self.trigger,*self.buttons.values()]:widget.setEnabled(ready)

    def choose_route(self,route):
        self.stop_mouse_hook()
        self.selected_route=route;self.translate.setChecked(route=='TRANSLATE');self.direct.setChecked(route=='DIRECT')
        if self.status.get('route') in {'TRANSLATE','DIRECT'}:self.client.send('activate',route=route)
        else:self.banner.setText(('已选择原始麦克风直通' if route=='DIRECT' else '已选择中文 → 日语翻译')+' · 按 Start 后才发送')

    def submit_quick(self,text,*,shortcut_id=None):
        self.quick_status.setText('Preparing · '+text)
        values=dict(text=text)
        if shortcut_id is not None:values['shortcut_id']=shortcut_id
        if not self.client.send('quick_reply',**values):self.quick_status.setText('Error · 控制队列已满（仅本地）')

    def action(self,name):
        if name=='recover' and self.status.get('runtime_state')=='STANDBY_READY' and (self.status.get('model_runtime') or {}).get('qwen',{}).get('state')=='QWEN_ERROR':
            self.client.send('qwen_recover');return
        if name in {'pause','stop','recover'}:self.stop_mouse_hook()
        if name in {'start','resume'}:self.client.send('activate',route=self.selected_route)
        else:self.client.send(name)

    def change_microphone(self):
        self.stop_mouse_hook()
        selected=self.microphones.currentData()
        if selected is not None:self.client.send('microphone',selection=str(selected))

    def change_voice(self):self.client.send('voice',preset=self.voices.currentData())

    def stop_mouse_hook(self):
        if self.mouse_hook:
            self.mouse_hook.close();self.mouse_hook=None
        self._hook_generation=None
    def toggle_boundary_arm(self):
        if not self.status.get('boundary',{}).get('armed',False):self._hook_failure=None
        self.client.send('boundary_arm',enabled=not self.status.get('boundary',{}).get('armed',False),generation=self.status.get('generation',0))
    def commit_boundary(self,cause,**values):
        if self.status.get('route')!='TRANSLATE' or not self.status.get('boundary',{}).get('armed'):return
        self.client.send('boundary_commit',generation=self.status.get('generation',0),at=time.perf_counter(),cause=cause,**values)
    def in_app_boundary(self,button,stamp):
        b=self.status.get('boundary',{})
        if b.get('settings',{}).get('scope')!='IN_APP_TRIGGER_AREA':return
        if b.get('armed'):self.client.send('boundary_commit',generation=self.status['generation'],at=stamp,cause='mouse',scope='IN_APP_TRIGGER_AREA',button=button)
    def on_global_button(self,button,stamp,generation):
        if generation!=self.status.get('generation') or not self.status.get('boundary',{}).get('armed'):return
        self.client.send('boundary_commit',generation=generation,at=stamp,cause='mouse',scope='GLOBAL_MOUSE',button=button)

    def open_settings(self):
        from .gui_settings import SettingsDialog
        if self.settings_dialog is None:self.settings_dialog=SettingsDialog(self.client,parent=self)
        else:
            # Discard un-applied edits when reopening, including before a reply.
            self.settings_dialog.load(self.settings_dialog.value)
            self.settings_dialog.load_asr_names(self.settings_dialog.asr_names_value)
        self.settings_dialog.apply.setEnabled(False)
        self.settings_dialog.show();self.settings_dialog.raise_();self.client.send('feature_get');self.client.send('asr_names_get')

    def on_message(self,message):
        self.camera_panel.message(message)
        if message.get('type')=='camera_settings' and isinstance(message.get('camera'),dict):
            self.status['camera']=message['camera']
        if self.settings_dialog:self.settings_dialog.message(message)
        if isinstance(message.get('status'),dict):self.update_status(message['status'])
        kind=message.get('type')
        if kind=='devices':
            current=self.status.get('microphone') or {};self.microphones.clear()
            for d in message['devices']:
                if d['is_physical_microphone'] and not d['is_cable_output_recording']:
                    self.microphones.addItem(f"{d['index']} · {d['name']} ({d['max_input_channels']} ch, {d['default_samplerate']} Hz)",d['index'])
            self.microphones.setCurrentIndex(self.microphones.findData(current.get('index')))
        elif kind=='error':self.on_error(message.get('message','未知错误'))
        elif kind=='quick_reply':self.quick_status.setText('Queued · '+message['text'])
        elif kind=='qwen_recover':self.translation_backend.setCurrentIndex(0);self.logs.appendPlainText('Qwen 显式恢复中；后端已回到 OPUS Fast。恢复完成后可重新选择 Qwen Fallback；个人 TTS 保持就绪。')
        elif kind=='boundary_commit' and not message.get('accepted'):
            self.logs.appendPlainText('仅本地分句状态：'+message.get('reason','未提交；继续收录'))
        elif kind=='command_done':self.logs.appendPlainText(f"完成：{message['command']}")

    def update_status(self,status):
        self.status=status;state=status['state'];ready=status.get('runtime_state')=='STANDBY_READY'
        self.state.setText('AI '+state);self.set_enabled(ready and not self.closing)
        self.camera_panel.set_ready(ready and not self.closing);self.camera_panel.update_state(status.get('camera'))
        qwen_failed=(status.get('model_runtime') or {}).get('qwen',{}).get('state')=='QWEN_ERROR'
        self.buttons['recover'].setEnabled(not self.closing and (state=='ERROR' or ready and qwen_failed and status.get('route')!='TRANSLATE'))
        route=status['route'];self.buttons['pause'].setEnabled(ready and route in {'TRANSLATE','DIRECT'})
        boundary=status.get('boundary',{});settings=boundary.get('settings',{});armed=boundary.get('armed',False)
        generation=status.get('generation',0);active=route=='TRANSLATE' and ready and boundary.get('effective_mode',settings.get('mode','AUTO'))!='AUTO'
        self.arm_button.setEnabled(active);self.arm_button.setText('DISARM 停用点击' if armed else 'ARM 启用点击')
        self.end_button.setEnabled(active and armed);self.trigger.setEnabled(active and armed)
        last=boundary.get('last') or {}
        self.boundary_info.setText(f"{settings.get('mode','AUTO')} / 生效 {boundary.get('effective_mode','AUTO')} · {'ARMED' if armed else 'DISARMED'} · {'收录中' if boundary.get('recording') else '等待语音'} {boundary.get('duration',0):.2f}s\n{settings.get('scope','IN_APP_TRIGGER_AREA')} · 上次 {last.get('cause','—')} · commit {last.get('committed_at','—')}")
        wants_hook=active and armed and settings.get('scope')=='GLOBAL_MOUSE'
        if self.mouse_hook and not self.mouse_hook.active:
            self.on_error('GLOBAL_MOUSE 已停止，仅本地提示；请重新明确 ARM。')
            self.stop_mouse_hook();self._hook_failure=generation;self.client.send('boundary_arm',enabled=False,generation=generation)
        if self.mouse_hook:self.boundary_info.setText(self.boundary_info.text()+f' · 鼠标队列丢弃 {self.mouse_hook.dropped}')
        if not wants_hook or self._hook_generation!=generation:self.stop_mouse_hook()
        if wants_hook and self.mouse_hook is None and self._hook_failure!=generation:
            try:
                from .mouse_hook import GlobalMouseHook
                self.mouse_hook=GlobalMouseHook(lambda button,stamp:self.global_button.emit(button,stamp,generation));self.mouse_hook.start();self._hook_generation=generation
            except Exception as exc:
                self.stop_mouse_hook();self._hook_failure=generation;self.on_error('GLOBAL_MOUSE 不可用，仅本地提示：'+str(exc));self.client.send('boundary_arm',enabled=False,generation=generation)
        self.buttons['resume'].setEnabled(ready and route=='PAUSED')
        if route=='DIRECT':
            self.banner.setText('● 正在发送原始麦克风声音 → CABLE Input · AI 保持热待机')
            self.banner.setStyleSheet('color:#9c4e00;font-weight:700')
        else:
            messages={'TRANSLATE':'● 正在翻译 · 仅个人日语合成音频发送到 CABLE Input',
                'PAUSED':'已暂停 · 两条路线均静音，模型保持热待机','STOPPED':'已停止 · 音频设备已释放，模型保留',
                'STANDBY':'AI 热待机就绪 · 尚未发送麦克风音频','ERROR':'错误 · 输出已静音，请检查日志后恢复'}
            self.banner.setText(messages.get(route,'正在加载并预热本地模型 · 尚未发送麦克风音频'))
            self.banner.setStyleSheet('color:#174b79')
        if status.get('diagnostic_fixture_input'):self.banner.setText('【诊断：忽略物理麦克风】'+self.banner.text())
        output=status.get('output') or {}
        self.output.setText(f"固定输出：{output.get('index','—')} · {output.get('name','CABLE Input（待验证）')} · {output.get('host_api','WASAPI')} · 会议软件选择 CABLE Output")
        self.input_meter.setValue(round(min(1,status.get('input_level',0))*100));self.output_meter.setValue(round(min(1,status.get('output_level',0))*100))
        runtime=status.get('model_runtime') or {};ids=runtime.get('model_ids') or {}
        self.models.setText(f"ASR / CPU MT / 个人 TTS：{'真实预热完成' if runtime.get('worker_alive') else '等待验证'} · 本窗口载入 {status.get('counts',{}).get('model_runtime_loads',runtime.get('model_loads',0))} 次 · 本实例预热 {runtime.get('warmup_runs',0)} 次 · TTS PID {ids.get('tts_pid','—')} · {LABELS.get(status.get('preset'),'—')}")
        qwen=runtime.get('qwen',{});self.models.setText(self.models.text()+f" · {qwen.get('state','QWEN_LOADING')} · Qwen请求 {qwen.get('requests',0)} · 载入 {qwen.get('total_model_loads',qwen.get('model_loads',0))} · 当前后端 {qwen.get('mode','OPUS_FAST')}")
        available=ready and route=='TRANSLATE' and status.get('translation_connected',False)
        for button in self.quick_buttons:
            button.setEnabled(available)
            button.setToolTip(button.text().split('：',1)[1]+' → VB-CABLE · '+status.get('preset','—')+(' · 可用' if available else ' · 需要活动 TRANSLATE'))
        self.backend_apply.setEnabled(ready and route not in {'TRANSLATE','DIRECT'})
        gpu=status.get('gpu')
        if gpu and 'used_mib' in gpu:self.gpu.setText(f"GPU 显存（整卡采样）：{gpu['used_mib']:.0f} / {gpu['total_mib']:.0f} MiB · {gpu['temperature_c']:.0f} °C")
        rows=status.get('utterances') or [];last=rows[-1] if rows else {};context=last.get('context') or {}
        quick=[r for r in rows if r.get('request_kind')=='QUICK_REPLY']
        if quick:self.quick_status.setText({'committed':'Queued','inference':'Synthesizing','buffered':'Queued / Playing','played':'Completed','cancelled':'Canceled','technical_error':'Error'}.get(quick[-1]['status'],quick[-1]['status'])+' · '+(quick[-1].get('japanese_text') or '快捷回复'))
        features=status.get('features')
        if features:
            self.speed.blockSignals(True);self.adaptive.blockSignals(True)
            self.speed.setValue(features['speed']);self.adaptive.setChecked(features['adaptive'])
            self.speed.blockSignals(False);self.adaptive.blockSignals(False)
            if self.settings_dialog:self.settings_dialog.value.update(speed=features['speed'],adaptive=features['adaptive'])
        tm=last.get('tts_metrics',{});seed=tm.get('sampling',{}).get('seed','—')
        if self.settings_dialog:self.settings_dialog.seed_status.setText(f'实际种子：{seed} · 预设：{status.get("preset","—")}')
        rejected=status.get('counts',{}).get('unsafe_translation_rejections',0)
        if last.get('status')=='translation_rejected' and last.get('id')!=getattr(self,'_last_rejected',None):
            self.logs.appendPlainText('翻译警告：UNRESOLVED_PRIVATE；修正仍不确定，本句无音频；可继续讲话。');self._last_rejected=last.get('id')
        audit=last.get('stage_times',{}).get('translation_policy')
        if audit and (last.get('id'),audit.get('outcome'))!=getattr(self,'_last_policy_log',None):
            self.logs.appendPlainText('仅本地：'+json.dumps(audit,ensure_ascii=False));self._last_policy_log=(last.get('id'),audit.get('outcome'))
        latency=last.get('first_output_callback_at',0)-context.get('speech_end_at',0)
        timing=f'{latency:.3f}s（软件回调）' if last.get('first_output_callback_at') and context.get('speech_end_at') else '—'
        stages=last.get('stage_times',{})
        backend=stages.get('translation_backend') or stages.get('translation_policy',{}).get('backend') or 'OPUS_FAST'
        def duration(start,end):return f'{end-start:.3f}s' if start is not None and end is not None and end>=start else '—'
        asr=duration(stages.get('asr_start'),stages.get('asr_end'));mt=duration(stages.get('mt_start'),stages.get('mt_end'))
        vad=duration(context.get('speech_end_at'),context.get('vad_commit_at'))
        tts_first=duration(stages.get('tts_start'),last.get('first_validated_pcm_at'))
        self.metrics.setText(f"缓冲 {status.get('queue_seconds',0):.2f}s · 等待 {status.get('waiting_jobs',0)} · 最新 {timing} · VAD {vad} / ASR {asr} / MT {mt} / TTS首PCM {tts_first} · 拦截 {rejected} · seed {seed} · {last.get('status','空闲')}")
        self.metrics.setText(self.metrics.text()+' · '+backend)
        self.metrics.setText(self.metrics.text()+f" · 队列 {status.get('waiting_jobs',0)}/{status.get('waiting_capacity',16)} · 最久等待 {status.get('oldest_waiting_age_seconds',0):.1f}s"+(' · ⚠ 等待队列接近满，请留出播放时间' if status.get('queue_nearly_full') else ''))
        rejection=status.get('queue_last_rejection')
        if rejection and rejection!=getattr(self,'_queue_last_rejection',None):
            self.logs.appendPlainText(f"仅本地：话语 {rejection['id']} 未接收（等待队列已满）；已接收话语继续按顺序处理。")
            self._queue_last_rejection=rejection
        if last.get('source_text'):self.source.setPlainText(last['source_text'])
        elif last.get('status')=='technical_error':self.source.setPlainText('【本句无有效识别文本】')
        if last.get('status')=='translation_rejected':self.japanese.setPlainText('【本句未播放】译文未通过商务敬体校验。')
        elif last.get('status')=='filler_skipped':self.japanese.setPlainText('FILLER_SKIPPED · 独立填充语，无音频。')
        elif last.get('status')=='technical_error':self.japanese.setPlainText('TECHNICAL_ERROR · 本句无音频，仅本地诊断。')
        elif last.get('japanese_text'):self.japanese.setPlainText(last['japanese_text'])
        elif last.get('source_text'):self.japanese.setPlainText('正在翻译 / 校正日语…（仅本地状态）')
        if ready and not self.requested_devices:self.requested_devices=True;self.client.send('devices')
        if status.get('error') and status['error']!=getattr(self,'_last_error',None):self.on_error(status['error']);self._last_error=status['error']
        self.status_received.emit(status)

    def on_error(self,text):self.logs.appendPlainText('错误：'+text)

    def on_exit(self,code):
        self.stop_mouse_hook()
        self.status.setdefault('boundary',{})['armed']=False
        self.close_allowed=True;self.set_enabled(False)
        if self.closing:self.close()
        else:self.state.setText(f'控制器已退出 ({code})');self.on_error('请关闭窗口后重新启动；没有自动切换设备或声音')

    def closeEvent(self,event):
        self.stop_mouse_hook()
        if self.close_allowed:
            if self.settings_dialog:self.settings_dialog.close()
            if self.camera_panel.dialog:self.camera_panel.dialog.close()
            for signal,slot in ((self.client.message,self.on_message),(self.client.failed,self.on_error),(self.client.exited,self.on_exit)):
                try:signal.disconnect(slot)
                except RuntimeError:pass
            event.accept();return
        event.ignore()
        if not self.closing:
            self.closing=True;self.set_enabled(False);self.state.setText('正在安全关闭…');self.client.shutdown()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preset',choices=NAMES,default=DEFAULT)
    p.add_argument('--port',type=int,default=8765);p.add_argument('--connect',help=argparse.SUPPRESS)
    p.add_argument('--fixture-input',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    app=QApplication(sys.argv[:1]);app.setApplicationName('VoiceBridge JP');app.setOrganizationName('VoiceBridgeJP')
    from .public_setup import ensure_reference
    if not ensure_reference():return 0
    window=MainWindow(**vars(args));window.show();return app.exec()


if __name__=='__main__':raise SystemExit(main())
