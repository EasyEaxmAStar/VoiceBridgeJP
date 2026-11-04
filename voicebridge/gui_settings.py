"""Additive settings dialog for the existing desktop shell; no inference on Qt."""
import copy, json
from pathlib import Path
from PySide6.QtCore import Qt,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog,QTabWidget,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QPushButton,
    QComboBox,QLineEdit,QSpinBox,QDoubleSpinBox,QCheckBox,QSlider,QTableWidget,QTableWidgetItem,QFileDialog,QPlainTextEdit,QScrollArea)
from .features import DEFAULTS,validate,ROOT
from .user_dictionary import load_json,load_csv,UserDictionary
from .variation import LEVELS
from ort.pronunciation import PronunciationLexicon


class SettingsDialog(QDialog):
    def __init__(self,client,value=None,parent=None):
        super().__init__(parent);self.client=client;self.value=copy.deepcopy(value or DEFAULTS)
        self.setWindowTitle('VoiceBridge JP · 高级设置');available=self.screen().availableGeometry()
        self.resize(min(1100,available.width()-32),min(760,available.height()-64))
        from .gui_theme import settings_theme
        settings_theme(self)
        layout=QVBoxLayout(self);self.tabs=QTabWidget();layout.addWidget(self.tabs)
        self.notice=QLabel('修改仅从后续话语生效；不重载 ASR、翻译或声线条件。DIRECT 始终保持原声。');self.notice.setWordWrap(True);layout.addWidget(self.notice)
        voice=QWidget();form=QFormLayout(voice);self.variation=QComboBox();self.variation.addItems(LEVELS)
        form.addRow('合成采样模式',self.variation)
        self.seed=QLineEdit();self.seed.setPlaceholderText('留空：自动选择；可填写0至2147483647的整数');form.addRow('调试种子锁',self.seed)
        self.seed_status=QLabel('实际种子：等待合成');form.addRow(self.seed_status)
        form.addRow(QLabel('官方基础模型使用STABLE；不包含个人种子研究或个人音色。'))
        self.tabs.addTab(voice,'声音变化')
        dictionary=QWidget();box=QVBoxLayout(dictionary);self.search=QLineEdit();self.search.setPlaceholderText('搜索用户条目');box.addWidget(self.search)
        language_row=QHBoxLayout();language_row.addWidget(QLabel('新增条目的源语言'))
        self.new_source_language=QComboBox();self.new_source_language.addItem('中文','zh');self.new_source_language.addItem('English','en');language_row.addWidget(self.new_source_language);language_row.addStretch();box.addLayout(language_row)
        self.table=QTableWidget(0,12);self.table.setHorizontalHeaderLabels(['中文 / 英文源词','日语译名（显示）','日语读音（平/片假名）','源语言','拼音（姓名ASR）','类别','别名（逗号或JSON）','区分大小写','启用','优先级','源匹配模式','备注'])
        for i,width in enumerate([150,180,200,90,150,140,150,80,60,70,100,180]):self.table.setColumnWidth(i,width)
        box.addWidget(self.table)
        row=QHBoxLayout();box.addLayout(row)
        for label,fn in [('新增',self.add_entry),('删除所选用户条目',self.delete_entry),('导入 JSON/CSV',self.import_dictionary),('导出 JSON/CSV',self.export_dictionary),('恢复覆盖层默认（空）',self.clear_dictionary)]:
            b=QPushButton(label);b.clicked.connect(fn);row.addWidget(b)
        self.preview=QLineEdit('Python と MySQL の接続を確認します。');box.addWidget(self.preview)
        info=QLabel('显示译名与合成读音独立；读音留空使用原规则。中文按词边界匹配，英文功能保留。姓名拼音提示仍由语音分段页ASR总开关控制。');info.setWordWrap(True);box.addWidget(info)
        self.spoken=QLabel();self.spoken.setTextFormat(Qt.PlainText);self.spoken.setWordWrap(True);box.addWidget(self.spoken)
        row=QHBoxLayout();box.addLayout(row)
        b=QPushButton('文本发音预览');b.clicked.connect(self.preview_text);row.addWidget(b)
        b=QPushButton('预览所选词条');b.clicked.connect(self.preview_selected_entry);row.addWidget(b)
        b=QPushButton('真实 TTS 测试（先停止 TRANSLATE）');b.clicked.connect(self.test_pronunciation);row.addWidget(b)
        self.tabs.addTab(dictionary,'用户词典')
        base=QWidget();box=QVBoxLayout(base);box.addWidget(QLabel('现有383条规则保持不变；用户显式覆盖优先。此表只读。'))
        self.base=QTableWidget(0,2);self.base.setHorizontalHeaderLabels(['原词','已验证读音']);self.base.setEditTriggers(QTableWidget.NoEditTriggers)
        baselex=PronunciationLexicon.load(ROOT/'config/pronunciation.ja.production.json')
        for _,term,reading in baselex.entries:
            n=self.base.rowCount();self.base.insertRow(n);self.base.setItem(n,0,QTableWidgetItem(term));self.base.setItem(n,1,QTableWidgetItem(reading))
        box.addWidget(self.base);self.tabs.addTab(base,'已验证词典')
        speech=QWidget();form=QFormLayout(speech);self.silence=QSlider(Qt.Horizontal);self.silence.setRange(300,1200)
        self.silence_label=QLabel();self.silence.valueChanged.connect(lambda x:self.silence_label.setText(f'{x} ms（30ms帧向下取整）'))
        form.addRow('句末静音',self.silence);form.addRow('',self.silence_label)
        self.minimum=QSpinBox();self.minimum.setRange(30,1000);self.minimum.setSuffix(' ms');form.addRow('最短实际语音',self.minimum)
        self.maximum=QDoubleSpinBox();self.maximum.setRange(3,30);self.maximum.setSuffix(' s');form.addRow('最长话语（超限安全停止）',self.maximum)
        self.energy=QDoubleSpinBox();self.energy.setDecimals(3);self.energy.setRange(.001,.1);self.energy.setSingleStep(.001);form.addRow('VAD RMS 能量门槛',self.energy)
        self.preroll=QSpinBox();self.preroll.setRange(0,600);self.preroll.setSuffix(' ms');self.preroll.setSingleStep(30);form.addRow('语音前缓冲（30ms帧）',self.preroll)
        self.smart=QCheckBox('最终中文 ASR 的完整句号断句批翻译（实验；不增加静音等待）');form.addRow(self.smart)
        form.addRow(QLabel('保留逗号、条件从句、数字和技术词。无实时增量 ASR。暂无独立后缓冲参数。'))
        row=QHBoxLayout()
        for label,fn in [('重置分段',self.reset_speech),('保存分段配置',self.export_profile),('恢复分段配置',self.import_profile)]:
            b=QPushButton(label);b.clicked.connect(fn);row.addWidget(b)
        form.addRow(row);self.tabs.addTab(speech,'语音分段')
        from ort.asr_vocabulary import DEFAULTS as NAME_DEFAULTS
        self.asr_names_value=copy.deepcopy(NAME_DEFAULTS)
        self.asr_names_enabled=QCheckBox('启用中文姓名 ASR 上下文（实验；不是 TTS 发音词典）')
        form.addRow(self.asr_names_enabled)
        self.asr_names_table=QTableWidget(0,3);self.asr_names_table.setHorizontalHeaderLabels(['中文姓名','普通话拼音','启用此条'])
        self.asr_names_table.setMinimumHeight(130);form.addRow(self.asr_names_table)
        names_row=QHBoxLayout()
        for label,fn in [('新增姓名',lambda:self.add_asr_name()),('删除姓名',self.delete_asr_names),('应用姓名词表',self.save_asr_names)]:
            b=QPushButton(label);b.clicked.connect(fn);names_row.addWidget(b)
        form.addRow(names_row)
        test_name=QPushButton('TEST NAME RECOGNITION · 选择普通话 WAV（不输出声音）')
        test_name.clicked.connect(self.test_name_recognition);form.addRow(test_name)
        self.asr_names_status=QLabel('ASR总开关控制本页与用户词典中的中文姓名提示；拼音可选。\n下一次识别生效，不重载权重，不执行无条件同音替换。')
        self.asr_names_status.setWordWrap(True);form.addRow(self.asr_names_status)
        self.load_asr_names(self.asr_names_value)
        boundary=QWidget();form=QFormLayout(boundary)
        self.boundary_mode=QComboBox();self.boundary_mode.addItems(['AUTO','MANUAL','HYBRID']);form.addRow('话语边界',self.boundary_mode)
        self.mouse_button=QComboBox();self.mouse_button.addItems(['RIGHT','LEFT','EITHER','DISABLED']);form.addRow('鼠标按钮',self.mouse_button)
        self.mouse_scope=QComboBox();self.mouse_scope.addItems(['IN_APP_TRIGGER_AREA','GLOBAL_MOUSE']);form.addRow('监听范围',self.mouse_scope)
        self.debounce=QSpinBox();self.debounce.setRange(100,1000);self.debounce.setSuffix(' ms');form.addRow('重复点击防抖',self.debounce)
        guidance=QLabel('AUTO 按静音提交；MANUAL 无需等待静音，点击结束当前句，继续收录下一句；HYBRID 取点击或静音中先到的边界。\n每次开始/切换路线后须在主窗口明确 ARM。GLOBAL_MOUSE 是可选的系统级监听，重启始终 DISARMED。\n全局点击也正常送达其他应用，可能打开右键菜单或误切句；建议先使用界面内触发区域。\n达到最长时限时 MANUAL/HYBRID 会显式提交并提示，切换/暂停/Stop/换麦克风则丢弃未提交录音。')
        guidance.setWordWrap(True);form.addRow(guidance)
        self.tabs.addTab(boundary,'鼠标分句')
        style=QWidget();box=QVBoxLayout(style);self.formality=QComboBox();self.formality.addItems(['STRICT_BUSINESS_POLITE','OFF']);box.addWidget(self.formality)
        box.addWidget(QLabel('默认严格商务敬体；未能安全确认的句子不合成，后续句子继续。此校验不保证覆盖全部日语。'))
        box.addWidget(QLabel('首选表达：完整中文话语 → 专业日语；仅匹配整句话。内置常用表达保留，用户设置优先。'))
        self.expressions=QTableWidget(0,2);self.expressions.setHorizontalHeaderLabels(['完整中文话语','首选日语敬体']);box.addWidget(self.expressions)
        row=QHBoxLayout();box.addLayout(row)
        b=QPushButton('新增首选表达');b.clicked.connect(lambda:self.add_expression());row.addWidget(b)
        b=QPushButton('删除所选表达');b.clicked.connect(self.delete_expression);row.addWidget(b)
        self.translation=QLineEdit('等一下');box.addWidget(self.translation)
        b=QPushButton('真实 CPU 翻译与敬体预览（先停止 TRANSLATE）');b.clicked.connect(lambda:self.client.send('translation_preview',text=self.translation.text()));box.addWidget(b)
        self.translation_result=QPlainTextEdit();self.translation_result.setReadOnly(True);self.translation_result.setMaximumHeight(160);box.addWidget(self.translation_result)
        self.tabs.addTab(style,'日语语体')
        row=QHBoxLayout();layout.addLayout(row)
        self.apply=QPushButton('应用并保存');self.apply.clicked.connect(self.save);row.addWidget(self.apply)
        b=QPushButton('取消 / 关闭');b.clicked.connect(self.close);row.addWidget(b)
        # Each populated page must remain reachable on smaller/high-DPI screens.
        pages=[(self.tabs.widget(i),self.tabs.tabText(i)) for i in range(self.tabs.count())]
        self.tabs.clear()
        for page,title in pages:
            scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(page);self.tabs.addTab(scroll,title)
        self.search.textChanged.connect(self.filter_entries);self.load(self.value)

    def load(self,value):
        self.value=copy.deepcopy(value);self.variation.setCurrentText(value['variation']);self.seed.setText('' if value['seed_lock'] is None else str(value['seed_lock']))
        self.formality.setCurrentText(value['formality']);self.expressions.setRowCount(0)
        boundary=value.get('boundary',DEFAULTS['boundary']);self.boundary_mode.setCurrentText(boundary['mode']);self.mouse_button.setCurrentText(boundary['button']);self.mouse_scope.setCurrentText(boundary['scope']);self.debounce.setValue(boundary['debounce_ms'])
        for source,target in value['expressions'].items():self.add_expression(source,target)
        self.load_speech(value['segmentation']);self.fill_dictionary(value['dictionary'])
    def load_speech(self,s):
        self.silence.setValue(s['silence_ms']);self.minimum.setValue(s['min_speech_ms']);self.maximum.setValue(s['max_seconds']);self.energy.setValue(s['energy']);self.preroll.setValue(s['pre_roll_ms']);self.smart.setChecked(s['smart_mandarin'])
        self.silence_label.setText(f"{s['silence_ms']} ms（30ms帧向下取整）")
    def add_asr_name(self,entry=None):
        entry=entry if isinstance(entry,dict) else dict(text='',pinyin='',enabled=True)
        row=self.asr_names_table.rowCount();self.asr_names_table.insertRow(row)
        for col,key in enumerate(['text','pinyin']):self.asr_names_table.setItem(row,col,QTableWidgetItem(entry[key]))
        item=QTableWidgetItem();item.setFlags(item.flags()|Qt.ItemIsUserCheckable);item.setCheckState(Qt.Checked if entry['enabled'] else Qt.Unchecked);self.asr_names_table.setItem(row,2,item)
    def delete_asr_names(self):
        for row in sorted({x.row() for x in self.asr_names_table.selectedIndexes()},reverse=True):self.asr_names_table.removeRow(row)
    def load_asr_names(self,value):
        self.asr_names_value=copy.deepcopy(value);self.asr_names_enabled.setChecked(value['enabled']);self.asr_names_table.setRowCount(0)
        for entry in value['entries']:self.add_asr_name(entry)
    def asr_names_settings(self):
        from ort.asr_vocabulary import validate as validate_names,context_for
        value=dict(schema=1,enabled=self.asr_names_enabled.isChecked(),entries=[dict(text=self.asr_names_table.item(r,0).text(),pinyin=self.asr_names_table.item(r,1).text(),type='person_name',enabled=self.asr_names_table.item(r,2).checkState()==Qt.Checked) for r in range(self.asr_names_table.rowCount())])
        validate_names(value);context_for(value);return value
    def test_name_recognition(self):
        path,_=QFileDialog.getOpenFileName(self,'选择已保存的普通话录音（最长15秒）','','WAV (*.wav)')
        if path:self.client.send('asr_name_test',path=path)
    def save_asr_names(self):
        try:self.client.send('asr_names_save',settings=self.asr_names_settings())
        except Exception as exc:self.asr_names_status.setText('姓名词表未应用：'+str(exc))
    def add_expression(self,source='',target=''):
        row=self.expressions.rowCount();self.expressions.insertRow(row)
        self.expressions.setItem(row,0,QTableWidgetItem(source));self.expressions.setItem(row,1,QTableWidgetItem(target))
    def delete_expression(self):
        for row in sorted({i.row() for i in self.expressions.selectedIndexes()},reverse=True):self.expressions.removeRow(row)
    def preferred_expressions(self):
        result={}
        for row in range(self.expressions.rowCount()):
            source=self.expressions.item(row,0).text();target=self.expressions.item(row,1).text()
            if source in result:raise ValueError('中文首选表达重复')
            result[source]=target
        return result
    def speech(self):return dict(silence_ms=self.silence.value(),min_speech_ms=self.minimum.value(),max_seconds=self.maximum.value(),energy=self.energy.value(),pre_roll_ms=self.preroll.value(),smart_mandarin=self.smart.isChecked())
    def reset_speech(self):self.load_speech(DEFAULTS['segmentation'])
    def fill_dictionary(self,data):
        self.table.setRowCount(0)
        for entry in data['entries']:self.add_entry(entry)
    def add_entry(self,entry=None):
        from .dictionary_schema import canonical
        legacy=isinstance(entry,dict) and 'source' in entry
        if legacy:
            entry={**dict(japanese=entry.get('source',''),reading='',aliases=[],case_sensitive=False,enabled=True,priority=100,match_mode='word',notes=''),**entry}
            entry=canonical(entry)
        else:entry={**dict(source_language=self.new_source_language.currentData(),source_term='',target_ja='',reading_kana='',pinyin='',entry_type='general_vocabulary',aliases=[],case_sensitive=False,enabled=True,priority=100,match_mode='word',notes=''),**(entry if isinstance(entry,dict) else {})}
        i=self.table.rowCount();self.table.insertRow(i)
        for col,key in enumerate(['source_term','target_ja','reading_kana','source_language','pinyin','entry_type','aliases','case_sensitive','enabled','priority','match_mode','notes']):
            if key in {'source_language','entry_type'}:
                combo=QComboBox()
                pairs=[('中文','zh'),('English','en')] if key=='source_language' else [('人名','person_name'),('技术词','technical_term'),('组织','organization'),('一般词汇','general_vocabulary'),('整句直出','direct_output')]
                for label,value in pairs:combo.addItem(label,value)
                combo.setCurrentIndex(combo.findData(entry[key]));self.table.setCellWidget(i,col,combo);continue
            value=entry[key];item=QTableWidgetItem(json.dumps(value,ensure_ascii=False) if isinstance(value,list) and any(',' in x for x in value) else ','.join(value) if isinstance(value,list) else str(value))
            if key in {'case_sensitive','enabled'}:
                item.setText('');item.setFlags(item.flags()|Qt.ItemIsUserCheckable);item.setCheckState(Qt.Checked if value is True else Qt.Unchecked)
            self.table.setItem(i,col,item)
        self.table.item(i,0).setData(Qt.UserRole,legacy)
        category=self.table.cellWidget(i,5)
        category.currentIndexChanged.connect(lambda _index,combo=category:self.sync_dictionary_match_mode(combo))
        self.sync_dictionary_match_mode(category)
    def sync_dictionary_match_mode(self,category):
        # Locate the current row by widget identity: deletions can move rows.
        for row in range(self.table.rowCount()):
            if self.table.cellWidget(row,5) is category:
                item=self.table.item(row,10)
                if category.currentData()=='direct_output':
                    item.setText('exact')
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setToolTip('整句直出必须完整匹配整句话，不会替换长句中的片段。')
                else:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    item.setToolTip('word：词语匹配；exact：整句精确匹配。')
                return
    def dictionary(self):
        entries=[]
        for row in range(self.table.rowCount()):
            text=lambda col:self.table.item(row,col).text()
            aliases=json.loads(text(6)) if text(6).lstrip().startswith('[') else [x.strip() for x in text(6).split(',') if x.strip()]
            e=dict(source_language=self.table.cellWidget(row,3).currentData(),source_term=text(0),target_ja=text(1),reading_kana=text(2),pinyin=text(4),entry_type=self.table.cellWidget(row,5).currentData(),aliases=aliases,case_sensitive=self.table.item(row,7).checkState()==Qt.Checked,enabled=self.table.item(row,8).checkState()==Qt.Checked,priority=int(text(9)),match_mode=text(10),notes=text(11))
            if e['entry_type']=='direct_output':e['match_mode']='exact'
            if self.table.item(row,0).data(Qt.UserRole) and e['source_language']=='en' and not e['pinyin'] and e['entry_type']=='technical_term':
                e=dict(source=e['source_term'],japanese=e['target_ja'] or e['source_term'],reading=e['reading_kana'],**{k:e[k] for k in ['aliases','case_sensitive','enabled','priority','match_mode','notes']})
            entries.append(e)
        schema=2 if any('source_term' in e for e in entries) else 1
        try:return load_json(json.dumps(dict(schema=schema,entries=entries),ensure_ascii=False).encode())
        except ValueError as exc:
            if str(exc)=='Pronunciation reading must be Japanese kana':
                raise ValueError('日语读音栏只能填写平假名或片假名（例如：たなかともうします）；汉字请放在日语译名栏，也可留空读音。') from exc
            raise
    def delete_entry(self):
        for row in sorted({x.row() for x in self.table.selectedIndexes()},reverse=True):self.table.removeRow(row)
    def clear_dictionary(self):self.table.setRowCount(0)
    def filter_entries(self,text):
        for row in range(self.table.rowCount()):self.table.setRowHidden(row,text.casefold() not in ' '.join(self.table.item(row,c).text() if self.table.item(row,c) else self.table.cellWidget(row,c).currentText() for c in range(12)).casefold())
    def fail(self,exc):self.notice.setText('未应用：'+str(exc))
    def preview_text(self):
        try:
            data=self.dictionary()
            if any(e.get('source_language')=='zh' for e in data['entries']):
                self.client.send('dictionary_preview',text=self.preview.text(),dictionary=data);return
            base=PronunciationLexicon.load(ROOT/'config/pronunciation.ja.production.json');spoken,audit=UserDictionary(data,base).apply(self.preview.text())
            self.spoken.setText(spoken+'\n匹配：'+', '.join(r['rule_id'] for r in audit['rules']))
        except Exception as exc:self.fail(exc)
    def preview_selected_entry(self):
        row=self.table.currentRow()
        if row>=0:self.preview.setText(self.table.item(row,1).text() or self.table.item(row,0).text());self.preview_text()
    def test_pronunciation(self):
        try:self.client.send('pronunciation_test',text=self.preview.text(),dictionary=self.dictionary())
        except Exception as exc:self.fail(exc)
    def import_dictionary(self):
        path,_=QFileDialog.getOpenFileName(self,'导入用户词典','','Dictionary (*.json *.csv)')
        if not path:return
        try:self.fill_dictionary((load_csv if Path(path).suffix.lower()=='.csv' else load_json)(Path(path).read_bytes()))
        except Exception as exc:self.fail(exc)
    def export_dictionary(self):
        try:
            data=self.dictionary();path,_=QFileDialog.getSaveFileName(self,'导出用户词典','','JSON (*.json);;CSV (*.csv)')
            if path:
                from .dictionary_schema import export_csv
                Path(path).write_bytes(export_csv(data) if Path(path).suffix.lower()=='.csv' else json.dumps(data,ensure_ascii=False,indent=2).encode('utf8'))
        except Exception as exc:self.fail(exc)
    def export_profile(self):
        path,_=QFileDialog.getSaveFileName(self,'保存分段配置','','JSON (*.json)')
        if path:Path(path).write_text(json.dumps(self.speech(),indent=2),encoding='utf8')
    def import_profile(self):
        path,_=QFileDialog.getOpenFileName(self,'恢复分段配置','','JSON (*.json)')
        if not path:return
        try:
            value=copy.deepcopy(self.value);value['segmentation']=json.loads(Path(path).read_text(encoding='utf8'));validate(value);self.load_speech(value['segmentation'])
        except Exception as exc:self.fail(exc)
    def save(self):
        try:
            value=copy.deepcopy(self.value);value.update(variation=self.variation.currentText(),seed_lock=int(self.seed.text()) if self.seed.text().strip() else None,dictionary=self.dictionary(),segmentation=self.speech(),formality=self.formality.currentText(),expressions=self.preferred_expressions(),boundary=dict(mode=self.boundary_mode.currentText(),button=self.mouse_button.currentText(),scope=self.mouse_scope.currentText(),debounce_ms=self.debounce.value()))
            validate(value)
            names=self.asr_names_settings()
            self.client.send('asr_names_save',settings=names)
            self.client.send('feature_save',settings=value)
            self.notice.setText('正在验证并保存；当前话语仍使用原快照。')
        except Exception as exc:self.fail(exc)

    def message(self,message):
        kind=message.get('type')
        if kind=='asr_names':
            self.load_asr_names(message['settings']);derived=message.get('effective',{}).get('derived_names',0)
            self.asr_names_status.setText(f"姓名上下文 {'启用' if message['settings']['enabled'] else '关闭'} · 本地提示{'可用' if message.get('supported',False) else '不可用'} · SHA {message['sha256'][:12]}\n用户词典贡献 {derived} 个姓名提示；独立姓名表优先。每次识别冻结设置，不替换识别结果。")
            effective=message.get('effective',{});independent={e['text']:e for e in message['settings']['entries']}
            active=[e for e in effective.get('settings',{}).get('entries',[]) if e['enabled']] if message['settings']['enabled'] else []
            details=[e['text']+'（'+('独立姓名表' if e['text'] in independent else '用户词典')+'）' for e in active]
            disabled=[e['text'] for e in independent.values() if not e['enabled']]
            last=message.get('last_recognition') or {}
            self.asr_names_status.setText(self.asr_names_status.text()+f"\n已应用/保存；有效 {len(active)} 个："+'、'.join(details)+'\n禁用独立条目覆盖（不会回退到词典）：'+'、'.join(disabled)+f"\n最近ASR：active={last.get('applied','未运行')} SHA={last.get('vocabulary_sha256','未运行')}")
        if kind=='asr_name_test':
            self.asr_names_status.setText(f"真实ASR：{message['text']}\n姓名命中={message['exact_name_present']} · 上下文={message['context_enabled']} · {message['seconds']:.3f}s\n请求SHA {message['context_sha256']}\n{message['context']}\n未发送到VB-CABLE")
        if kind=='features':
            self.load(message['settings']);self.apply.setEnabled(True);self.notice.setText(f"已应用 · SHA {message['sha256'][:12]} · 保存 {message.get('reload_seconds',0)*1000:.2f} ms")
        elif kind=='translation_preview':
            audit=message['audit'];self.translation_result.setPlainText('译文候选（含术语处理）：'+audit['baseline_translation']+'\n'+('已拦截：本句不播放。' if message.get('rejected') else '敬体结果：'+message['text']))
        elif kind=='dictionary_preview':self.spoken.setText(message['spoken']+'\n匹配：'+', '.join(r['rule_id'] for r in message['audit']['rules']))
        elif kind=='pronunciation_test':
            self.notice.setText('已生成本地发音测试 WAV（按当前保音高速度）；不会发送到 VB-CABLE。')
            QDesktopServices.openUrl(QUrl.fromLocalFile(message['wav']))
        elif kind=='error':self.fail(message.get('message','操作失败'))
