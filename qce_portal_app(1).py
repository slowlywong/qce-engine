# -*- coding: utf-8 -*-
import streamlit as st
import time, random, re, hashlib, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ------------------------------
# QCE ENGINE (Rule-based, text-derived)
# ------------------------------

THAI_LOWER = lambda s: s  # Thai has no case; keep original for matching substrings

HEDGES = set([
    'อาจ','น่าจะ','เหมือน','คิดว่า','คง','ลอง','ถ้า','หรือเปล่า','ไหม','หรือไม่','บางที','ประมาณ',
    'หวังว่า','เชื่อว่า',
    'maybe','might','perhaps','seems','seem','sort of','kind of','try','if','whether','hope','believe'
])

ASSERTIVES = set([
    'ฉันรู้','รู้','พร้อม','ชัดเจน','แน่ชัด','ตั้งใจ','ต้องการ','ยืนยัน','แน่นอน','ขออนุญาต','เห็นชัด','ชัดแจ้ง','พร้อมแล้ว','รับรู้','ยอมรับ','ตกลง',
    'i know','know','ready','clearly','definitely','certainly','intend','intention','want','confirm','agree'
])

NEG_CONFLICT = set([
    'ไม่แน่ใจ','ไม่มั่นใจ','ไม่ชัดเจน','ไม่พร้อม','ลังเล','สับสน','กลัว','กังวล','ขัดแย้ง','ตีกัน','ไม่แน่','ไม่รู้','ไม่เข้าใจ','ไม่อยาก'
])

THETA_TOKENS = set([
    'นิ่ง','แก่น','เป็นหนึ่งเดียว','วงกลม','ภายใน','สภาวะ','ความหมาย','เงียบ','รู้โดยไม่พูด','ไม่ต้องพูด','ความจริง','ชัดแจ้ง','ศูนย์กลาง','หนึ่งเดียว'
])

GAMMA_TOKENS = set([
    'พร้อม','ยืนยัน','ตกลง','เริ่ม','ทำเลย','ขอ','รับรอง','ตรง','พุ่ง','รู้ทันที','เดี๋ยวนี้','รับรู้','ชัดเจน','ประกาศ'
])

ALPHA_TOKENS = set(['สงบ','เบา','สบาย','นิ่งๆ','ช้า','ผ่อน'])
BETA_TOKENS  = set(['เพราะ','ดังนั้น','เหตุผล','วิเคราะห์','ตรรกะ','โครงสร้าง','ขั้นตอน','ข้อเท็จจริง'])
DELTA_TOKENS = set(['พัก','ล้า','เหนื่อย','เจ็บ','ช้า','ฟื้น','หลับ','หยุดพัก'])

WAVE_MULTIPLIER = {
    'alpha': 1.00,
    'beta': 0.95,
    'theta': 1.05,
    'delta': 0.90,
    'gamma': 1.10,
}


def contains_any(text: str, vocab: set) -> bool:
    return any(tok in text for tok in vocab)


def qce_read(texts, debug=False):
    """
    QCE reading from short conversational snippets (1–3 lines).
    texts: list[str] — last 1–3 lines (e.g., assistant line + user line)
    returns: dict with intent, discordance, waves, consent_score, status, reasons
    """
    raw = '\n'.join([t.strip() for t in texts if t and t.strip()])
    text = THAI_LOWER(raw)

    # Feature flags
    f = {
        'has_hedge': contains_any(text, HEDGES),
        'has_assertive': contains_any(text, ASSERTIVES),
        'has_neg_conflict': contains_any(text, NEG_CONFLICT),
        'has_theta': contains_any(text, THETA_TOKENS),
        'has_gamma': contains_any(text, GAMMA_TOKENS),
        'has_alpha': contains_any(text, ALPHA_TOKENS),
        'has_beta': contains_any(text, BETA_TOKENS),
        'has_delta': contains_any(text, DELTA_TOKENS),
    }

    # Additional structural signals
    char_len = len(text)
    short_utterance = char_len <= 30  # short & sharp → gamma assist
    very_short = char_len <= 15

    # Intent estimation (0..1)
    intent = 0.5
    if f['has_assertive']: intent += 0.18
    if f['has_theta']: intent += 0.15
    if f['has_gamma']: intent += 0.15
    if f['has_alpha']: intent += 0.05
    if f['has_beta']:  intent += 0.03
    if f['has_hedge']: intent -= 0.12
    if f['has_neg_conflict']: intent -= 0.20
    if short_utterance: intent += 0.05
    if very_short and f['has_gamma']: intent += 0.03
    intent = max(0.0, min(1.0, intent))

    # Discordance estimation (0..1) — lower is better
    discord = 0.30
    if f['has_hedge']: discord += 0.20
    if f['has_neg_conflict']: discord += 0.25
    if f['has_assertive']: discord -= 0.15
    if f['has_theta']: discord -= 0.12
    if 'ไม่ต้องพูด' in text: discord -= 0.08
    if f['has_beta'] and not f['has_theta'] and not f['has_gamma']:
        discord += 0.05  # over-analysis without core
    # Clamp
    discord = max(0.0, min(1.0, discord))

    # Wave scoring
    wave_scores = {
        'alpha': 0.0,
        'beta': 0.0,
        'theta': 0.0,
        'gamma': 0.0,
        'delta': 0.0,
    }
    if f['has_alpha']: wave_scores['alpha'] += 0.6
    if f['has_beta']:  wave_scores['beta']  += 0.6
    if f['has_theta']: wave_scores['theta'] += 0.7
    if f['has_gamma']: wave_scores['gamma'] += 0.7
    if f['has_delta']: wave_scores['delta'] += 0.6

    # Short & sharp boosts gamma; metaphor/abstraction boosts theta
    if short_utterance: wave_scores['gamma'] += 0.1
    if contains_any(text, set(['วงกลม','แก่น','สภาวะ','หนึ่งเดียว','เงียบ'])):
        wave_scores['theta'] += 0.1

    # Pick present waves = scores >= 0.6 (allow multiple)
    present_waves = [w for w,s in wave_scores.items() if s >= 0.6]
    if not present_waves:
        # default to alpha (calm) if none detected
        present_waves = ['alpha']

    # Wave multiplier = average if multiple
    wm = float(np.mean([WAVE_MULTIPLIER[w] for w in present_waves]))

    # Consent score
    ni = max(0.0, intent - discord)
    coh = 1.0 - abs(intent - (1.0 - discord))
    cs = (0.7 * ni + 0.3 * coh) * wm
    cs = float(max(0.0, min(1.0, cs)))

    if cs >= 0.75:
        status = 'Consent Granted'
    elif cs >= 0.50:
        status = 'Needs Clarification'
    else:
        status = 'Consent Denied'

    # Reasons (Explainability)
    reasons = []
    if f['has_assertive']:
        reasons.append('พบสัญญาณยืนยัน/ตั้งใจ → หนุน Intent')
    if f['has_hedge']:
        reasons.append('พบถ้อยคำลังเล/เงื่อนไข → เพิ่ม Discordance')
    if f['has_neg_conflict']:
        reasons.append('พบสัญญาณความไม่มั่นใจ/ตีกันภายใน')
    if f['has_theta']:
        reasons.append('ถ้อยคำชี้แก่น/สภาวะ → จัดเป็น THETA')
    if f['has_gamma'] or short_utterance:
        reasons.append('ถ้อยคำคม/สั้น/ประกาศสภาวะ → จัดเป็น GAMMA')
    if f['has_alpha']:
        reasons.append('โทนสงบ/สบาย → สนับสนุน ALPHA')
    if f['has_beta']:
        reasons.append('การอธิบายเชิงเหตุผล → สนับสนุน BETA')

    # Hash for privacy-preserving logging
    input_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    result = {
        'input_hash': input_hash,
        'raw_text': raw,
        'intent': round(intent, 3),
        'discordance': round(discord, 3),
        'waves': present_waves,
        'wave_multiplier': round(float(wm), 3),
        'consent_score': round(cs, 3),
        'status': status,
        'reasons': reasons,
    }
    if debug:
        result['wave_scores'] = wave_scores
        result['char_len'] = char_len
    return result


# ------------------------------
# STREAMLIT APP (Merged with original UI spirit)
# ------------------------------

st.set_page_config(page_title='QCE Portal (Merged)', layout='wide')

# Header
col1, col2 = st.columns([3,1])
with col1:
    st.title('QCE Portal — Conversational Intent Reader')
    st.subheader('Quantum-Emotional Consent Engine (Merged with Bitr Pass UI)')
with col2:
    st.markdown("""<div style='text-align:right; font-size:13px;'>
        <b>Status:</b> Prototype · <b>Mode:</b> Text-derived<br>
        <b>Privacy:</b> Hash-only logging
    </div>""", unsafe_allow_html=True)

st.markdown('---')

# Sidebar: instructions
with st.sidebar:
    st.header('วิธีใช้งาน (Instructions)')
    st.markdown("""
    1) ป้อนบทสนทนาสั้น ๆ 1–3 บรรทัด (เช่น ประโยคจากระบบ + ประโยคจากคุณ)
    2) กด **Run QCE Reading** เพื่อวิเคราะห์ Intent / Discordance / Wave / Consent
    3) หากพอใจผลลัพธ์ ให้กด **Add to Timeline** เพื่อบันทึกลง Session
    4) สามารถ **Export Logs** เป็น CSV ได้

    **หมายเหตุสำคัญ**: ผลลัพธ์เป็นการประเมินเชิงสัญลักษณ์เพื่อการสะท้อนตนเองและการสื่อสาร ไม่ใช่หลักฐานทางกฎหมาย/การแพทย์
    """)

# Input area
st.header('Authentication / Reading Sequence')
col_in1, col_in2 = st.columns(2)
with col_in1:
    a_line = st.text_input('บรรทัดจากระบบ/ผู้อ่าน (ออปชัน)', value='')
with col_in2:
    u_line = st.text_input('บรรทัดจากคุณ/ผู้ใช้ (สำคัญ)', value='')

extra = st.text_area('บรรทัดเพิ่มเติม (ออปชัน, 0–1 บรรทัด)', height=60, value='')

run = st.button('Run QCE Reading', type='primary')

if 'timeline' not in st.session_state:
    st.session_state.timeline = []  # list of dicts

if run:
    # Progress animation (visual only)
    prog = st.progress(0)
    status_text = st.empty()
    for p in range(100):
        time.sleep(0.01 + random.random()*0.01)
        prog.progress(p+1)
        if p < 25:
            status_text.text('Scanning conversational cues...')
        elif p < 55:
            status_text.text('Extracting intent & discordance signals...')
        elif p < 85:
            status_text.text('Mapping wave harmonics...')
        else:
            status_text.text('Finalizing consent evaluation...')

    texts = [a_line, u_line]
    if extra.strip():
        texts.append(extra.strip())
    result = qce_read(texts)

    st.success('Reading Complete')

    # Metrics row
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric('Intent', f"{result['intent']:.2f}")
    with mc2:
        st.metric('Discordance', f"{result['discordance']:.2f}")
    with mc3:
        st.metric('Consent Score', f"{result['consent_score']:.2f}")
    with mc4:
        st.metric('Wave Multiplier', f"{result['wave_multiplier']:.2f}")

    # Waves & status
    st.subheader('Waves detected')
    st.write(', '.join([w.upper() for w in result['waves']]))

    st.subheader('สถานะ (Decision)')
    st.info(result['status'])

    # Reasons
    st.subheader('เหตุผล (Explainability)')
    for r in result['reasons']:
        st.markdown(f'- {r}')

    # Add to timeline button
    if st.button('Add to Timeline (บันทึกผลรอบนี้)'):
        entry = {
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'input_hash': result['input_hash'],
            'intent': result['intent'],
            'discordance': result['discordance'],
            'consent_score': result['consent_score'],
            'waves': '|'.join(result['waves']),
            'status': result['status'],
        }
        st.session_state.timeline.append(entry)
        st.success('เพิ่มลงไทม์ไลน์เรียบร้อยแล้ว')

# Timeline & Export
st.markdown('---')
st.header('Wave Timeline & Logs')

if len(st.session_state.timeline) == 0:
    st.warning('ยังไม่มีข้อมูลในไทม์ไลน์ กด Run และ Add to Timeline ก่อน')
else:
    df = pd.DataFrame(st.session_state.timeline)
    st.dataframe(df, use_container_width=True)

    # Wave Timeline heatmap (ALPHA..DELTA)
    rounds = [f"รอบที่ {i+1}" for i in range(len(df))]
    waves = ['alpha','beta','theta','gamma','delta']
    presence = np.zeros((len(df), len(waves)))
    for i, ws in enumerate(df['waves']):
        present = set(ws.split('|'))
        for j, w in enumerate(waves):
            presence[i, j] = 1.0 if w in present else 0.0

    fig = plt.figure(figsize=(9, 4))
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    cmap = ListedColormap([[0.92,0.92,0.92],[0.0,0.6,0.6]])
    ax1.imshow(presence.T, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax1.set_title('ไทม์ไลน์คลื่นพลังงาน (Wave Timeline)', fontsize=12)
    ax1.set_yticks(np.arange(len(waves)))
    ax1.set_yticklabels([w.upper() for w in waves])
    ax1.set_xticks(np.arange(len(rounds)))
    ax1.set_xticklabels(rounds, rotation=45)
    ax1.set_xticks(np.arange(-0.5, len(rounds), 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(waves), 1), minor=True)
    ax1.grid(which='minor', color='white', linewidth=1.2)

    for i in range(len(rounds)):
        for j in range(len(waves)):
            if presence[i, j] == 1:
                ax1.text(i, j, '✓', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

    # Consent score line
    ax2 = plt.subplot2grid((1, 3), (0, 2))
    scores = df['consent_score'].tolist()
    ax2.plot(range(1, len(scores)+1), scores, marker='o', color='#0c7c59', linewidth=2)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlim(0.8, len(scores)+0.2)
    ax2.set_xticks(range(1, len(scores)+1))
    ax2.set_xticklabels(rounds, rotation=90, fontsize=8)
    ax2.set_title('คะแนนการยินยอม (Consent)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.4)
    for x, y in enumerate(scores, start=1):
        ax2.text(x, min(0.97, y+0.03), f"{y:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    # Export logs
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button('Export Logs (CSV)', data=buf.getvalue(), file_name='qce_logs.csv', mime='text/csv')

# Footer note
st.markdown("""<div style='font-size:12px; color:#666; padding-top:12px;'>
    © QCE Prototype — ค่าที่ได้เป็นการประเมินเชิงสัญลักษณ์เพื่อการสะท้อนตนเองและการสื่อสาร ไม่ใช่หลักฐานทางกฎหมาย/การแพทย์
</div>""", unsafe_allow_html=True)
