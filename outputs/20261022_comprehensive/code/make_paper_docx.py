from pathlib import Path
import json
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = Path(__file__).resolve(); BASE = HERE.parents[1]; RES = BASE/'results'; ASSET = BASE/'paper_assets'; OUT = BASE/'雷达抗干扰算法实验部分.docx'
def font(run, size=10.5, bold=False, white=False):
    run.font.name='Times New Roman'; run._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); run.font.size=Pt(size); run.bold=bold
    if white: run.font.color.rgb=RGBColor(255,255,255)
def shade(cell, fill):
    p=cell._tc.get_or_add_tcPr(); e=OxmlElement('w:shd'); e.set(qn('w:fill'),fill); p.append(e)
def table(doc, headers, rows):
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
    for i,h in enumerate(headers):
        shade(t.rows[0].cells[i],'1F4E79'); t.rows[0].cells[i].text=''; r=t.rows[0].cells[i].paragraphs[0].add_run(str(h)); font(r,8.5,True,True); t.rows[0].cells[i].paragraphs[0].alignment=WD_ALIGN_PARAGRAPH.CENTER
    for ri,row in enumerate(rows):
        cells=t.add_row().cells
        for i,v in enumerate(row):
            if ri%2: shade(cells[i],'EAF2F8')
            cells[i].text=''; p=cells[i].paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run(str(v)); font(r,8.5); cells[i].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    doc.add_paragraph()
def para(doc, text, indent=True):
    p=doc.add_paragraph(); p.paragraph_format.line_spacing=1.35; p.paragraph_format.space_after=Pt(6)
    if indent: p.paragraph_format.first_line_indent=Pt(24)
    r=p.add_run(text); font(r); return p
def image(doc, name, caption):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(ASSET/name),width=Inches(6.15))
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run(caption); font(r,9); p.paragraph_format.space_after=Pt(8)

doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.72);sec.bottom_margin=Inches(.72);sec.left_margin=Inches(.82);sec.right_margin=Inches(.82)
for n,s in [('Normal',10.5),('Title',18),('Heading 1',14),('Heading 2',12)]:
    st=doc.styles[n]; st.font.name='Times New Roman'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'黑体' if n!='Normal' else '宋体');st.font.size=Pt(s);st.font.color.rgb=RGBColor(0,0,0)
p=doc.add_paragraph(style='Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;r=p.add_run('雷达抗干扰决策算法实验与结果分析');font(r,18,True)
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;r=p.add_run('基于信号级 SINR 仿真 稠密 Oracle 与鲁棒性评估');font(r,11)
para(doc,'本节围绕小样本条件下的雷达抗干扰动作推荐问题，介绍实验数据、参数设置、评价指标及主要实验结果。算法接收干扰类型、载频、带宽、功率及欺骗参数等结构化观测，输出雷达载频、带宽、脉宽、增益和波形类型等抗干扰动作，并以信号级仿真得到的 SINR 衡量推荐效果。')

doc.add_heading('1 实验任务与数据集',1)
para(doc,'训练集与留出集均按干扰场景组织，同一场景内的多条记录共享干扰状态而对应不同雷达动作。训练标签为信号级仿真 SINR。为减少历史固定 50 场景评估带来的测试集适配风险，本节将后 450 个未进入历史主报告的留出场景作为配对统计的主报告对象。')
table(doc,['数据集','场景数','每场景动作数','样本数','用途'],[['信号级训练集','5,000','20','100,000','训练与检索库'],['完整留出集','500','10','5,000','泛化评估'],['新主统计集','450','10','4,500','配对统计检验'],['稠密 Oracle 集','250','动态搜索','-','绝对真 regret']])
para(doc,'留出集包含宽带压制、近距离压制、窄带压制、随机、扫频、间歇采样和多假目标共 7 类干扰；各类场景数为 65 至 78 个，类别分布相对均衡。')

doc.add_heading('2 实验设置与评价指标',1)
doc.add_heading('2.1 比较方法与参数设置',2)
para(doc,'比较方法包括 Random、Q-only、Retrieve+Q、Retrieve+Q-Union、Q-Ensemble-Mean 及 Q-Ensemble 悲观打分。Retrieve+Q 在状态潜空间搜索相似训练场景，再从其历史动作中选择 Q 值最大的动作；Union 在检索动作基础上加入随机探索动作，使策略同时利用经验先验与动作覆盖。Q-Ensemble 由 5 个不同随机种子训练的 Q 网络组成，分别以预测均值 μ 或悲观得分 μ-λσ 进行候选排序。')
table(doc,['参数项','设置'],[['训练场景数','5,000'],['每训练场景动作数','20'],['留出场景数','500，主统计为后 450'],['检索 Top-K','默认 5，扫描 1、3、5、10、20'],['随机候选数','默认 100，扫描 5、20、50、100、500'],['Ensemble 成员数','5'],['悲观系数 λ','默认 1.0，扫描 0 至 3.0'],['JSR','默认 10 dB，测试范围 0 至 20 dB']])
doc.add_heading('2.2 指标与统计检验',2)
para(doc,'历史兼容指标为 regret vs Oracle-K10，即留出场景中 10 个随机动作的最大 SINR 减去推荐动作 SINR；负值表示推荐动作超过 K10 弱参照。为构造绝对指标，对 250 个新场景逐场景随机采样 5,000 个动作并进行坐标细化，将所得最高 SINR 作为稠密近似 Oracle，进而计算真 regret。该 Oracle 为真最优的下界，因此真 regret 为真实 regret 的上界。')
para(doc,'统计检验采用逐场景配对比较，报告均值、中位数、p90、p95 和 CVaR-10%。均值差异使用 10,000 次 bootstrap、100,000 次符号置换检验与 Wilcoxon 检验，并采用 Holm 校正处理多重比较。')

doc.add_heading('3 主结果与统计分析',1)
core=json.loads((RES/'core_500'/'summary.json').read_text(encoding='utf-8')); names={'q_only':'Q-only','retrieve_q':'Retrieve+Q','retrieve_q_union':'Retrieve+Q-Union','ensemble_ctx_mean':'Ensemble-Mean','ensemble_ctx_pessimistic':'Ensemble 悲观 λ=1'}
rows=[]
for m in core['methods']:
    s=m['fresh450'];rows.append([names[m['method']],f"{s['mean']:.2f}",f"{s['median']:.2f}",f"{s['p90']:.2f}",f"{s['p95']:.2f}",f"{s['cvar10']:.2f}"])
table(doc,['方法','均值','中位数','p90','p95','CVaR-10%'],rows)
para(doc,'上表为后 450 个新场景的 regret vs Oracle-K10，数值越小越好。Q-only 的平均 regret 为 4.24 dB；引入检索后，Retrieve+Q 降至 -3.58 dB；将检索池与随机候选池合并后，Union 进一步降至 -4.08 dB，说明检索经验与探索候选具有互补性。')
para(doc,'Ensemble-Mean 的均值为 -4.25 dB，p95 为 1.92 dB，优于 Union 的 -4.08 dB 与 2.73 dB。与此同时，相对 Union 的平均 SINR 提升为 0.17 dB，95% bootstrap 区间为 [-0.09, 0.46] dB，Holm 校正后的置换检验 p=0.698。故应将 Ensemble 的作用界定为降低尾部风险，而非平均性能的显著提升。')

doc.add_page_break()
doc.add_heading('4 稠密 Oracle 下的绝对性能',1)
true=pd.read_csv(RES/'true_regret_fresh250.csv')
table(doc,['方法','平均真 regret','中位数','p90','p95','CVaR-10%'],[[names[r.method],f'{r.mean_true_regret:.2f}',f'{r.median_true_regret:.2f}',f'{r.p90:.2f}',f'{r.p95:.2f}',f'{r.cvar10:.2f}'] for _,r in true.iterrows()])
image(doc,'fig_true_regret.png','图 1 250 个新场景上的平均真 regret 比较')
para(doc,'稠密搜索共调用 940,761 次仿真，耗时约 66.5 分钟。稠密 Oracle 的平均 SINR 为 25.06 dB，而同一批场景 K10 弱参照的均值仅为 18.25 dB，表明有限随机候选会明显低估部分场景的优化空间。Ensemble-Mean 的平均真 regret 为 4.38 dB，为比较方法中的最低值；Union 为 4.52 dB。两者绝对差距较小，与配对统计检验结论一致。')

doc.add_heading('5 候选机制消融与跨条件测试',1)
para(doc,'候选动作覆盖是当前性能的关键因素。Top-K 从 5 增至 20 后，后 450 场景的平均 regret 从 -4.24 dB 改善至 -4.77 dB；随机候选数由 100 增至 500 后，平均 regret 进一步改善至 -5.01 dB，p95 降至 0.85 dB。这说明现有系统的主要改进空间在于候选池覆盖，而非继续增加网络结构复杂度。')
image(doc,'fig_candidate_sweeps.png','图 2 检索深度和随机候选池规模的消融结果')
para(doc,'悲观系数扫描中，λ=0.5 的平均 regret 最低，为 -4.27 dB；λ 在 0 至 1 时差异较小，而 λ 大于 1.5 后性能逐步退化。因此，不确定性惩罚宜作为弱约束使用。')
image(doc,'fig_jsr.png','图 3 冻结策略在不同 JSR 条件下的性能')
para(doc,'在 JSR=0 至 20 dB 的跨条件测试中，平均 regret 保持在 -4.29 至 -4.17 dB；p95 regret 由 1.74 dB 上升到 2.17 dB。该结果表明冻结策略具有一定跨干扰强度稳定性，但不等同于多 JSR 联合训练的泛化结论。')

doc.add_page_break()
doc.add_heading('6 鲁棒性实验',1)
para(doc,'鲁棒性实验在全部 5,000 个训练场景上实施。对干扰载频、增益、带宽、扫频步进、间歇采样时长与延迟、假目标数等激活参数施加扰动，每个半径采用 20 个随机方向并加入逐坐标边界搜索，以近似最坏情况。')
rob=pd.read_csv(RES/'robust_formal'/'summary_by_radius.csv')
table(doc,['扰动半径','平均最坏退化','中位数','p95','CVaR-10%','动作翻转率'],[[f'{r.rho*100:.0f}%',f'{r.mean_worst_degradation:.2f}',f'{r.median_worst_degradation:.2f}',f'{r.p95_worst_degradation:.2f}',f'{r.cvar10_worst_degradation:.2f}',f'{r.any_flip_rate*100:.1f}%'] for _,r in rob.iterrows()])
image(doc,'fig_robustness.png','图 4 干扰参数扰动下的最坏情况退化与决策翻转风险')
para(doc,'在 10% 扰动半径下，平均最坏退化为 1.83 dB，p95 为 8.18 dB，57.0%的场景在至少一个扰动方向上出现最优动作翻转；20% 扰动下翻转率升至 62.4%。因此，鲁棒性不能仅依据随机扰动均值判断，后续应以最坏情况 regret、CVaR 或风险敏感目标训练鲁棒策略。')

doc.add_heading('7 小结',1)
para(doc,'实验验证了检索先验与随机探索候选并集的有效性，并将历史 50 场景评估扩展至 500 场景。稠密 Oracle 结果显示，当前方法距近似最优仍有约 4.4 dB 的平均差距，特别是在间歇采样、随机和多假目标等局部结构复杂的干扰场景中存在进一步优化空间。若不立即重训，推荐将推理配置调整为 Top-K=20、随机候选数=500，并采用 Ensemble-Mean 或弱悲观 λ≈0.5。正式替换默认配置前，应在封存的独立最终测试集上复核。')
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER;r=footer.add_run('雷达抗干扰项目 实验部分');font(r,8)
doc.save(OUT); print(OUT)
