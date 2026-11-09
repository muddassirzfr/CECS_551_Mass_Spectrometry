"""Create a corrected copy of the Phase 3 PDF, preserving its original EDA figures.

Run with PYTHONPATH pointing to temporary reportlab/PyMuPDF dependencies if needed.
The source PDF is never overwritten. Numeric statements come from recorded results.
"""
from html import escape
import json
from pathlib import Path
import re

import pymupdf as fitz
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Flowable,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path('/Users/muddassirzafar/Downloads/Phase 3 Prelimary Data Analysis and Feasibility.pdf')
RECORDS = ROOT/'casmi_generative_ranker/benchmark_results'
TEMP = ROOT/'tmp/pdfs'
OUTPUT = ROOT/'output/pdf/Phase_3_Preliminary_Data_Analysis_and_Feasibility_Revised.pdf'
TEMP.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def ascii_text(text):
    for a, b in {'\u2013':'-', '\u2014':'-', '\u2011':'-', '\u2192':' -> ',
                 '\u2264':'<=', '\u00d7':'x', '\u2019':"'", '\u2018':"'",
                 '\u201c':'"', '\u201d':'"', '\u200b':''}.items():
        text = text.replace(a, b)
    return text


STYLES = {
    'Body': ParagraphStyle('Body', fontName='Times-Roman', fontSize=10.5, leading=13.1, spaceAfter=8),
    'Title': ParagraphStyle('Title', fontName='Times-Bold', fontSize=15, leading=18, spaceAfter=9, keepWithNext=True),
    'Heading': ParagraphStyle('Heading', fontName='Times-Bold', fontSize=11.5, leading=14, spaceBefore=5, spaceAfter=5, keepWithNext=True),
    'Caption': ParagraphStyle('Caption', fontName='Times-Roman', fontSize=9.5, leading=11.5, spaceAfter=8),
    'Cell': ParagraphStyle('Cell', fontName='Times-Roman', fontSize=8.1, leading=9.6),
    'CellHead': ParagraphStyle('CellHead', fontName='Times-Bold', fontSize=8.5, leading=10.2, textColor=colors.white),
}


def para(text, kind='Body', markup=False):
    text = ascii_text(text)
    return Paragraph(text if markup else escape(text), STYLES[kind])


def write_pdf(name, story, pagesize=letter, margin=72, one_page=False):
    path = TEMP/name
    doc = SimpleDocTemplate(str(path), pagesize=pagesize, leftMargin=margin,
                            rightMargin=margin, topMargin=margin, bottomMargin=margin)
    doc.build(story)
    if one_page and len(fitz.open(path)) != 1:
        raise RuntimeError(f'{name} must fit one replacement page')
    return path


class ModelDiagram(Flowable):
    def __init__(self):
        super().__init__()
        self.width = 468
        self.height = 192

    def draw(self):
        c = self.canv
        width, gap = 87, 8
        def row(y, title, labels, shade):
            c.setFillColor(colors.HexColor('#244f68'))
            c.setFont('Helvetica-Bold', 8.5)
            c.drawString(0, y+48, title)
            for index, text in enumerate(labels):
                x = index*(width+gap)
                c.setStrokeColor(colors.HexColor('#607e96'))
                c.setFillColor(colors.HexColor(shade))
                c.roundRect(x, y, width, 39, 4, fill=1)
                c.setFillColor(colors.HexColor('#203347'))
                lines = text.split('\n')
                c.setFont('Helvetica', 7.5)
                for j, line in enumerate(lines):
                    c.drawCentredString(x+width/2, y+25-(j*9), line)
                if index < len(labels)-1:
                    c.setStrokeColor(colors.HexColor('#536473'))
                    start = x+width
                    c.line(start, y+19, start+gap, y+19)
                    c.line(start+gap, y+19, start+gap-3, y+21)
                    c.line(start+gap, y+19, start+gap-3, y+17)
        row(127, 'Baseline: CNN -> RNN (GRU), separately benchmarked',
            ['Spectrum\n127 peaks + precursor', 'Fixed m/z grid\n2 x 40,960 bins', 'CNN encoder\n32 / 64 / 128',
             'GRU decoder\n2 layers, width 128', 'SMILES\nup to 160 tokens'], '#eef1f8')
        row(54, 'Primary model: one Transformer encoder-decoder',
            ['Spectrum + metadata\n129 total positions', 'Transformer encoder\n4 layers, width 384', 'Causal decoder\n4 layers, 160-token cap',
             'Candidate SMILES\nbeam width 25', 'Ranked output\nup to 25 guesses'], '#eaf4f1')
        c.setDash(3, 2)
        c.setStrokeColor(colors.HexColor('#a68433'))
        c.setFillColor(colors.HexColor('#fff7e6'))
        c.roundRect(147, 0, 321, 38, 3, fill=1)
        c.setDash()
        c.setFillColor(colors.HexColor('#5b4c28'))
        c.setFont('Helvetica', 7.7)
        c.drawCentredString(307.5, 24, 'Optional: ranker, ensemble, tuning or other extensions')
        c.drawCentredString(307.5, 12, 'Only with held-out MRR@25 gain and remaining allocation')


def overlay_paragraphs(source, page_number, items):
    """Remove complete text regions, then place vector text without touching figures."""
    page = source[page_number-1]
    for rect, _, _ in items:
        page.add_redact_annot(fitz.Rect(rect), fill=(1, 1, 1))
    page.apply_redactions(images=0, graphics=0)
    overlay = TEMP/f'overlay-{page_number}.pdf'
    c = canvas.Canvas(str(overlay), pagesize=letter)
    for rect, text, kind in items:
        x0, y0, x1, y1 = rect
        paragraph = para(escape(ascii_text(text)).replace('\n\n','<br/><br/>'), kind, markup=True)
        _, height = paragraph.wrap(x1-x0, y1-y0)
        if height > y1-y0:
            raise RuntimeError(f'Overlay on page {page_number} exceeds its text region')
        paragraph.drawOn(c, x0, 792-y0-height)
    c.save()
    with fitz.open(overlay) as content:
        page.show_pdf_page(page.rect, content, 0)


def main():
    source = fitz.open(SOURCE)
    projections = json.loads((RECORDS/'projections.json').read_text())
    cnn = json.loads((RECORDS/'cnn_gru.json').read_text())
    transformer = json.loads((RECORDS/'transformer.json').read_text())
    base = projections['base_plan_gpu_hours']
    minimum = projections['minimum_total_budget_with_reserve_and_prior_spend']
    remaining = projections['proposed_remaining_gpu_hours_after_base_and_reserve']

    overlay_paragraphs(source, 8, [
        ((72, 313, 544, 419),
         '6. Implications on Model Parameters\n\nSpectrum EDA scanned 2,539,608 spectra and sampled 20,000 per split. At 128 spectral positions (127 fragment peaks plus precursor), about 90% of sampled spectra retain at least 98% of summed intensity (Figure 10). The implemented Transformer adds one metadata token: 129 total encoder positions. Its decoder limit is 160 tokens including BOS/EOS.', 'Body'),
        ((72, 648, 544, 705),
         'Figure 10. Per-spectrum intensity retained versus the spectral-position budget for training and validation samples. The plotted budget includes the precursor and excludes the separate metadata token. Thus 128 spectral positions correspond to 129 total Transformer encoder positions.', 'Caption')
    ])
    note = 'Figure 11 reports the original supplied targets and historical 80/20 split. The benchmark uses competition-normalized targets, the saved three-way split, and a 160-token decoder cap.'
    # This note is below the original table, leaving the plot and all values unchanged.
    note_page = source[8]
    note_box = fitz.Rect(72, 725, 540, 749)
    if note_page.insert_textbox(note_box, ascii_text(note), fontsize=8.0, fontname='tiro', color=(.25,.25,.25)) < 0:
        raise RuntimeError('Historical EDA note does not fit')

    original_norm = ' '.join(source[9].get_text().split())
    original_norm = original_norm[original_norm.index('Normalization')+len('Normalization'):].strip()
    continuation = source[10].get_text().split('Adduct and Collision-Energy Conditioning')[0]
    original_norm += ' '+' '.join(continuation.split())
    page10 = write_pdf('replacement-10.pdf', [
        para('Spectrum Cleaning and Filtering', 'Heading'),
        para('The computational pilot removes nonfinite and nonpositive fragment m/z or intensity values, retains the 127 most intense fragment peaks, and includes one precursor position. Relative-intensity filtering below 1% of the base peak and removal of peaks above precursor mass remain proposed experiments; they were not used to produce the benchmark timings. Spectra with very few peaks will be investigated separately.'),
        para('The retained spectrum therefore has at most 128 spectral positions. The Transformer also receives a separate metadata token, for a maximum of 129 total encoder positions. The EDA result that about 90% of sampled spectra preserve at least 98% of summed intensity applies to the 128-position spectral budget; it does not count metadata.'),
        para('Segmentation and Input Representation', 'Heading'),
        para('Each retained peak contains its m/z value and normalized intensity. Variable-length spectra are padded within each batch, and padding masks prevent padded positions from contributing as real spectral measurements. The Transformer embeds each mass/intensity pair and uses self-attention to relate fragment peaks, including peaks far apart on the m/z axis.'),
        para('For the CNN-to-RNN baseline, the same retained peaks are placed on a fixed m/z grid before convolution. The pilot uses 0.1-m/z bins over [0, 4096), with separate channels for fragment intensities and a precursor indicator. This avoids applying local convolution to an intensity-sorted peak list. Masses outside the configured range produce an explicit error rather than silently being clipped.'),
        para('The encoder representations differ by model: the Transformer has at most 129 input positions, whereas the CNN operates on two channels of 40,960 mass bins and then pools its features. Both generators share a training-only BPE tokenizer and a maximum of 160 SMILES tokens, including BOS and EOS. Longer targets are excluded explicitly; generated prefixes without EOS are rejected.'),
        para('Normalization', 'Heading'), para(original_norm),
    ], one_page=True)

    page11 = write_pdf('replacement-11.pdf', [
        para('Adduct and Collision-Energy Conditioning', 'Heading'),
        para('Mass spectra can vary with the ion adduct and collision energy even when the underlying molecule is identical. Adducts such as [M+H]+ and [M+Na]+ shift the measured precursor m/z and may change fragmentation. The EDA found that [M+H]+ accounts for approximately 53.1% of spectra. Treating these conditions as inputs can help the model distinguish measurement effects from molecular-structure differences; a predictive benefit has not yet been established.'),
        para('The computational pilot supplies adduct, ionization-mode and instrument categories and a collision-energy summary, with missingness handled explicitly. The Transformer combines these features in a separate metadata token, bringing its spectral-plus-metadata input cap to 129 positions. The CNN-to-GRU baseline instead adds metadata to its pooled context before initializing the recurrent decoder. Both generators use the same conditioning fields.'),
        para('Collision energy governs the extent of fragmentation: higher energy often produces more and smaller fragments, while lower energy may preserve larger fragments. Available collision-energy measurements are summarized for conditioning; unavailable values remain explicitly missing. Stable categorical hashing avoids fitting a vocabulary on held-out data, but hash collisions are possible. Performance and per-adduct robustness must be measured on the identity-disjoint validation partition.'),
        para('Augmentation', 'Heading'),
        para('No data augmentation is currently planned. The dataset contains approximately 2.54 million spectra, and the first comparison will use existing measurements and the shared preprocessing pipeline. If validation reveals poor generalization for molecules or adducts with few examples, augmentation may be investigated later, subject to the remaining budget.'),
    ], one_page=True)

    stages = [
        ['Stage', 'Model', 'What it does', 'Rationale'],
        ['1', 'CNN -> RNN (GRU)', 'Convolves a fixed m/z grid; a recurrent decoder writes SMILES.', 'Required baseline. Separately measured; prediction quality remains untested.'],
        ['2', 'Transformer encoder-decoder', 'Self-attention over 129 total encoder positions; causal decoding up to 160 tokens.', 'Relates distant fragments and conditions each output token on the spectrum.'],
        ['3', 'Ranker / ensemble (optional)', 'Re-scores generated candidates or combines validated models.', 'Proceed only with held-out MRR@25 improvement and remaining GPU allocation.'],
        ['4', 'Potential extensions', 'Tuning, attention variants or chemistry-informed training.', 'Defer until the two core models and evaluation work within budget.'],
    ]
    stage_table = Table([[para(value, 'Cell') for value in row] for row in stages], colWidths=[30,112,169,157])
    stage_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d6e7f3')),
        ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#596a78')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
        ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    page12 = write_pdf('replacement-12.pdf', [
        para('Model Plan', 'Title'),
        para('Build the planned CNN-to-RNN baseline and one Transformer encoder-decoder as the main model. The concrete pilot instantiates the RNN as a GRU. Pooled peak embeddings plus GRU is retained only as an auxiliary compute probe and is excluded from the required model comparison and budget. A ranker or ensemble is conditional on validated improvement and remaining resources.'),
        Spacer(1,8), ModelDiagram(), Spacer(1,10),
        para('Figure 12. Planned CNN-to-RNN baseline and primary Transformer pipeline. The dashed box denotes optional extensions; it does not commit to an ensemble. Both generators use the same 160-token SMILES cap.', 'Caption'),
        stage_table,
    ], one_page=True)

    page13 = write_pdf('replacement-13.pdf', [
        para('1. Baseline: CNN -> RNN', 'Heading'),
        para('The planned baseline remains a CNN-to-RNN sequence-to-sequence model. A CNN encodes local spectral patterns on a fixed m/z grid, and a GRU decoder generates SMILES one token at a time. This supplies a reference for evaluating whether the Transformer justifies its additional compute cost. Local convolution may miss fragment relationships spanning distant masses, while a recurrent hidden state may make long-range SMILES dependencies harder to represent; the comparison must be tested rather than assumed.'),
        para('The measured pilot uses Conv1D channels 32, 64 and 128, kernel size 9 and stride 4 in each layer; adaptive average pooling to 16 cells; a 128-dimensional projection; and a two-layer width-128 GRU decoder. Metadata conditions the initial state. These are concrete baseline settings, not details specified by the original broad model plan. The separate CNN pilot measured 2,552 spectra/second; pooled-GRU timings are not substituted for it.'),
        para('2. Proposed Model: Transformer Encoder-Decoder', 'Heading'),
        para('The encoder receives 127 fragment peaks, one precursor position and one metadata token: 129 total encoder positions. The decoder generates BPE-tokenized SMILES with a limit of 160 tokens including BOS/EOS, using cross-attention to the encoded spectrum at each step. The pilot uses four encoder and four decoder layers, width 384, eight heads, feed-forward width 1536, dropout 0.1, and pre-LayerNorm.'),
        para('Self-attention allows peaks to interact across m/z distances and can help relate fragments that originate from the same molecular substructure. Causal decoder attention can represent ring and branch dependencies while cross-attention conditions each token on the spectrum. The encoder processes peak representations in parallel. These mechanisms motivate the model; prediction quality and superiority over the baseline remain untested.'),
        para('3. Optional Ranker or Ensemble', 'Heading'),
        para('A single Transformer is the primary commitment. A ranker or additional generators are optional and require useful candidate recall, improvement in held-out MRR@25, and sufficient allocation after the evaluation/retry reserve. The initial ranker cost estimate covers at most 2,000 reserved structures with up to five spectra each (10,000 spectra), not all 247,342 reserved spectra. Generating candidates for that full partition would require about 10.01 GPU compute-hours, plus CPU processing and unmeasured full-holdout fitting.'),
        para('4. Potential Extensions', 'Heading'),
        para('Hyperparameter tuning, efficient attention, and chemistry-informed losses remain stretch goals. If an estimated additional cost exceeds the remaining budget, retain the single Transformer and defer the optional stage. Appendix A gives preliminary cost projections for the separately measured CNN baseline and Transformer.'),
    ], one_page=True)

    page16 = write_pdf('replacement-16.pdf', [
        para('Feasibility Risks and Mitigation', 'Title'),
        para('The short computational pilots support preliminary compute-feasibility estimates. They do not demonstrate prediction quality, full-dataset loading speed, or full-dataset host-memory feasibility. Both the CNN baseline and Transformer need structure-level validation before any claim about novel-molecule generalization.'),
        para('Full-Dataset Loading and RAM', 'Heading'),
        para('The timed loop cycles 4,096 prepared spectra. The original 2.19 GiB host-RAM peak belongs only to that small-sample PBS job, not a full pandas load of 2,539,608 spectra. The current training loader reads the full Parquet file into memory. Before a production run, profile the full load on an allocated compute node with a RAM limit, or implement streaming. The added 10% loading allowance is an assumption and cannot establish full-data behavior.'),
        para('Representation, Rare Motifs and Generalization', 'Heading'),
        para('Shared spectral motifs may help learning, but rare adducts, heavy atoms and large molecules can be poorly represented. Preserve identity-disjoint training, validation and ranker partitions; fit tokenization only on training targets; evaluate molecule-level MRR@25 and candidate recall; and report per-adduct sample counts so noisy subgroup metrics are identifiable. Use SMILES sequences as targets; InChIKey identity is used for matching and deduplication, not as a sequence target to interpolate.'),
        para('Invalid or Missing Candidate Coverage', 'Heading'),
        para('Autoregressive decoding may produce invalid or unfinished SMILES. Require EOS, parse with RDKit, normalize consistently and deduplicate by scoring identity. Measure valid unique coverage and candidate recall before adding a ranker, since ranking cannot recover a correct molecule absent from the generated candidates.'),
        para('Budget Overrun and Optional Stages', 'Heading'),
        para(f'The recalculated CNN + GRU/Transformer base plan is approximately {base:.2f} GPU allocation-hours for ten epochs per model plus the stated validation/test decoding. A proposed total of 12 GPU-hours reserves two additional hours for extra evaluation and partial retries. Including 0.060 hours already used, the minimum is {minimum:.2f} hours and only {remaining:.2f} hours remain outside the reserve. The capped ranker would require about 0.56 allocation-hours with inline CPU processing, so it is deferred under this proposal.'),
        para('The confirmed 1.5-hour allowance applies to benchmarking only. A full-training allocation has not been specified or approved. The two-hour reserve cannot cover a full repeated seven-hour Transformer run; full retries or broader evaluation require a revised allocation. Appendix A records configurations, workload scopes, estimates and the underlying evidence bundle.'),
    ], one_page=True)

    methods = [
        para('Appendix A. Computational Benchmark and Resource Plan', 'Title'),
        para('Evidence Status and Hardware', 'Heading'),
        para('Prediction quality remains untested. The pilots support preliminary compute-feasibility estimates, not a full-dataset feasibility demonstration. One NVIDIA GeForce RTX 3090 was allocated per job using PyTorch 2.7.1+cu118 and CUDA 11.8, BF16 autocast with FP32 model/optimizer state, eight reserved CPU cores and four PyTorch/BLAS threads. CPU ranker fitting used one thread. The original successful job was 159855; the separate CNN job was 159856.'),
        para('Configurations and Timing Procedure', 'Heading'),
        para('Transformer: four encoder/four decoder layers, width 384, eight heads, feed-forward width 1536, dropout 0.1, pre-LayerNorm, vocabulary 512 and 17,614,208 parameters. Input: 129 total positions (127 fragments + precursor + metadata). Output: at most 160 SMILES tokens including BOS/EOS. CNN + GRU: two channels of 40,960 mass bins on [0, 4096) at 0.1 m/z, Conv1D channels 32/64/128, kernel 9, stride 4, pooling to 16 cells, a width-128 context and two GRU layers; 771,744 parameters. Metadata is fused into the recurrent initial state.'),
        para('Each model used microbatch 32 and four-way accumulation (effective batch 128), 20 warm-up optimizer updates and 150 timed full updates, processing 19,200 spectra by cycling a 4,096-spectrum sample. Timing includes loading/tokenization, device transfer, forward pass, backward pass, gradient clipping and AdamW update; CUDA was synchronized around the interval. Dynamic padding was used. Sampled training target lengths had median 12, P95 24 and maximum 69 tokens. The separate full-cap check measures memory for one optimizer update, not sustained worst-case throughput.'),
        para('Beam timing used 64 validation spectra, inference batch eight, beam width 25 and no KV cache. Candidate sampling used 16 reserved spectra, 100 samples per spectrum, inference batch eight and chunk 100. The same train-only tokenizer, normalized targets, identity partitions, peak retention and metadata were used. Recheck decoding after obtaining a validated checkpoint because generated length and candidate validity will change.'),
        para('Loading and Memory Limitation', 'Heading'),
        para('The full partition counts are 2,242,315 training, 49,951 validation and 247,342 reserved spectra. Timing a repeatedly reused 4,096-spectrum subset and observing 2.19 GiB peak host RAM in the original job does not establish full-Parquet loading behavior. Full-data RAM and I/O remain unmeasured. The current training loader reads the whole Parquet file; production use requires a compute-node full-load profile under a memory limit or streaming. GPU memory below is peak allocated memory; reserved-memory values remain available in the raw JSON records.'),
        para('Memory is stage-specific: the CNN candidate-sampling stage recorded 0.37 GiB peak allocated and 10.77 GiB peak reserved GPU memory. Its 0.155 GiB training stress-check allocation does not bound inference reservation. Both figures come from the separate CNN raw record.', 'Caption'),
    ]
    methods_pdf = write_pdf('appendix-methods.pdf', methods)

    raw_rows = [line for line in (RECORDS/'report.md').read_text().splitlines() if line.startswith('| ')]
    cells = []
    for index, line in enumerate(raw_rows):
        values = [value.strip().replace('**','').replace('NVIDIA GeForce RTX 3090','RTX 3090') for value in line.strip('|').split('|')]
        cells.append([para(value, 'CellHead' if index==0 else 'Cell') for value in values])
    resource_table = Table(cells, colWidths=[135,52,70,140,162,153], repeatRows=1)
    resource_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#244f68')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0f4f7')]),
        ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#9aa9b3')),
        ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),5),
        ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    table_pdf = write_pdf('appendix-resource-table.pdf', [
        para('Appendix A. Resource Table', 'Title'),
        para('All total times are extrapolations; CPU fitting is a synthetic timing proxy. Alternative scopes replace their subset row and are not additive. Pooled-GRU is auxiliary and excluded from the base plan.', 'Caption'),
        resource_table,
        Spacer(1,8), para('The base plan uses the separately measured CNN + GRU baseline and one Transformer. GPU compute time excludes CPU processing; GPU allocation time includes CPU work when the GPU remains reserved. Prediction quality remains untested.', 'Caption'),
    ], pagesize=landscape(letter), margin=40)

    budget_pdf = write_pdf('appendix-budget.pdf', [
        para('Appendix A. Cost Assumptions and Decision Rule', 'Title'),
        para('Training Projection', 'Heading'),
        para('Training hours = training spectra x planned epochs / measured spectra per second / 3600. The scenario uses 2,242,315 spectra x ten epochs per required model, plus measured validation and checkpoint costs, a 10% additional loading allowance and preparation time. It predicts 2.76 hours for CNN + GRU and 7.16 hours for the Transformer. These are extrapolations from short pilots; no full training has been completed.'),
        para('Validation Scope', 'Heading'),
        para('The training estimates include 97 teacher-forced token-loss validation events, capped at 5,000 spectra each and extrapolated from the 553-spectrum sample. The base plan separately includes one beam evaluation of the same 553 spectra for each required model and Transformer decoding of the 1,213 test spectra. It does not include beam decoding the full 49,951-spectrum validation partition after every epoch. One full Transformer validation-plus-test beam pass would take about 0.297 GPU-hours at the measured rate.'),
        para('Ranker Scope and Allocation Accounting', 'Heading'),
        para('The 0.40 GPU compute-hour ranker estimate covers at most 2,000 structures x five spectra = 10,000 spectra, plus about 0.14 CPU chemistry-hours. Sampling all 247,342 reserved spectra would instead require about 10.01 GPU compute-hours and 3.39 CPU chemistry-hours; full-holdout fitting is unmeasured. The 58.8-second CPU fixture fits 70% and then all of 2,000 synthetic groups with 500 candidates each, 250 trees and 15 leaves; its synthetic labels provide no quality evidence. Retaining the GPU through subset CPU chemistry and this fitting proxy would cost about 0.56 allocation-hours. Full-holdout inline generation would cost about 13.42 allocation-hours before fitting.'),
        para('Actual Allowance and Proposed Training Budget', 'Heading'),
        para(f'Only the measurement allowance of 1.5 hours is confirmed. GPU allocation usage across the original job, its startup retry and the CNN job totals 216 seconds (0.060 hours). The corrected CNN + GRU/Transformer base plan is {base:.2f} hours. Adding a two-hour reserve for extra evaluation/partial retries and the recorded measurement use gives {minimum:.2f} hours. A proposed 12-hour total leaves {remaining:.2f} hours outside the reserve; this is a proposal, not an approved full-training allocation.'),
        para('If an optional stage costs more than the remaining allocation, retain the single Transformer and defer that stage. The capped ranker does not fit the proposed spare budget; an ensemble is unmeasured and deferred. Any later ranker or ensemble also requires improved held-out MRR@25. The earlier 9.60/11.65-hour totals included pooled-GRU, not the planned CNN baseline, and are superseded.'),
        para('Supporting Evidence', 'Heading'),
        para('The accompanying benchmark_evidence.zip supplies raw model JSON records, hardware and sample manifests, PBS output/accounting, package versions, source snapshots and checksums. A labeled copy clarifies the CNN raw record\'s inherited peak-position counter: the CNN actually uses 40,960 grid bins per channel. Timing values are unchanged. The original input PDF and EDA plots remain preserved; figures on pages 8 and 9 are historical EDA, not prediction-quality results.'),
    ])

    corrected = fitz.open()
    replacements = {9:page10, 10:page11, 11:page12, 12:page13, 15:page16}
    for index in range(len(source)):
        if index in replacements:
            with fitz.open(replacements[index]) as replacement:
                corrected.insert_pdf(replacement)
        else:
            corrected.insert_pdf(source, from_page=index, to_page=index)
    for path in (methods_pdf, table_pdf, budget_pdf):
        with fitz.open(path) as appendix:
            corrected.insert_pdf(appendix)
    for number, page in enumerate(corrected, 1):
        page.insert_text((page.rect.width-77, page.rect.height-24), f'{number} / {len(corrected)}',
                         fontsize=8, fontname='helv', color=(.35,.4,.44))
    corrected.set_metadata(dict(title='Phase 3 - Preliminary Data Analysis and Feasibility (Revised)',
                                author='The Overfitters', subject='CNN-RNN baseline, Transformer configuration and preliminary compute estimates'))
    evidence = ROOT/'output/benchmark_evidence.zip'
    if not evidence.is_file():
        raise RuntimeError('Create the evidence ZIP before building the final PDF')
    corrected.embfile_add('benchmark_evidence.zip', evidence.read_bytes(),
                         filename='benchmark_evidence.zip', desc='Raw timing records, PBS logs, source snapshots and checksums')
    corrected.save(OUTPUT, garbage=4, deflate=True)
    checked = fitz.open(OUTPUT)
    if checked.embfile_get('benchmark_evidence.zip') != evidence.read_bytes():
        raise RuntimeError('Embedded evidence attachment differs from delivered ZIP')
    texts = [page.get_text() for page in checked]
    all_text = '\n'.join(texts)
    for stale in ('up to 128 tokens', 'maximum encoder input length of 128 tokens'):
        if stale in all_text:
            raise RuntimeError(f'Stale active architecture description: {stale}')
    for expected in ('129 total encoder positions', '160', 'CNN', '11.99', '9.93'):
        if expected not in all_text:
            raise RuntimeError(f'Missing expected correction: {expected}')
    changed_pages = [8,9,10,11,12,13,16]+list(range(17,len(checked)+1))
    for number in changed_pages:
        checked[number-1].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(TEMP/f'revised-{number}.png')
    print(json.dumps(dict(output=str(OUTPUT), pages=len(checked), changed_pages=changed_pages), indent=2))


if __name__ == '__main__':
    main()
