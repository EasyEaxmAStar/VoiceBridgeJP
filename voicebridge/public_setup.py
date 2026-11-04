"""First-run static reference setup for the official pretrained TTS model."""
from pathlib import Path
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QLabel,QLineEdit,QPlainTextEdit,
    QPushButton,QFileDialog,QDialogButtonBox,QMessageBox,QComboBox)


def ensure_reference():
    from ort.config import Config
    from ort.pipeline.gptsovits_reference import validate_reference
    from .camera_settings import atomic_json
    cfg=Config();error=''
    try:
        validate_reference(cfg.gptsovits_ref_audio,cfg.gptsovits_ref_text,'ja')
        import json,os
        local=Path(__file__).resolve().parents[1]/'config/public_tts.local.json'
        if os.environ.get('VOICEBRIDGE_INPUT_DEVICE') or (local.is_file() and json.loads(local.read_text(encoding='utf8')).get('microphone')):return True
    except Exception as exc:error=str(exc)
    dialog=QDialog();dialog.setWindowTitle('VoiceBridge JP · 官方 TTS 参考配置');dialog.resize(620,410)
    layout=QVBoxLayout(dialog)
    label=QLabel('使用官方 GPT-SoVITS v2ProPlus 预训练权重。\n请选择你有权使用的日语参考 WAV，并填写音频中实际说出的准确文本。\n不附带固定说话人，不使用开发者的录音或个人权重。');label.setWordWrap(True);layout.addWidget(label)
    path=QLineEdit(cfg.gptsovits_ref_audio);layout.addWidget(path)
    browse=QPushButton('选择日语参考 WAV');layout.addWidget(browse)
    def choose():
        selected,_=QFileDialog.getOpenFileName(dialog,'选择参考音频','','WAV (*.wav)')
        if selected:path.setText(selected)
    browse.clicked.connect(choose)
    microphone=QComboBox();microphone.addItem('请选择物理麦克风','')
    try:
        from .audio_devices import enumerate_devices
        for device in enumerate_devices():
            if device.host_api=='Windows WASAPI' and device.is_physical_microphone:microphone.addItem(device.name,device.name)
    except Exception as exc:note_error=str(exc)
    layout.addWidget(QLabel('物理麦克风（使用明确设备名称，不自动替换）'));layout.addWidget(microphone)
    text=QPlainTextEdit();text.setPlaceholderText('准确日语逐字稿；建议使用3–10秒、清晰自然且无截断的录音');text.setPlainText(cfg.gptsovits_ref_text);layout.addWidget(text)
    note=QLabel(error);note.setWordWrap(True);layout.addWidget(note)
    buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons)
    def save():
        try:
            reference=validate_reference(path.text().strip(),text.toPlainText().strip(),'ja')
            selected=microphone.currentData()
            if not selected:raise ValueError('请选择物理麦克风；不能静默替换设备')
            atomic_json(Path(__file__).resolve().parents[1]/'config/public_tts.local.json',dict(schema=1,reference_wav=reference.path,reference_text=reference.text,reference_language='ja',microphone=selected))
            # Explicit environment reference overrides have priority. Do not
            # pretend a saved file supersedes conflicting startup overrides.
            import os
            os.environ['ORT_GPTSOVITS_REF_AUDIO']=reference.path
            os.environ['ORT_GPTSOVITS_REF_TEXT']=reference.text
            os.environ['ORT_GPTSOVITS_REF_LANGUAGE']='ja'
            os.environ['VOICEBRIDGE_INPUT_DEVICE']=selected
            dialog.accept()
        except Exception as exc:note.setText('未保存：'+str(exc))
    buttons.accepted.connect(save);buttons.rejected.connect(dialog.reject)
    return dialog.exec()==QDialog.Accepted
