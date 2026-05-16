# -*- coding: utf-8 -*-
"""
QCE Engine Test - สามารถรันใน VS Code ได้เลย
ไม่ต้องใช้ Streamlit - สมบูรณ์ทั้งหมด โดยไม่ตัดรายละเอียดสำคัญ
"""
import hashlib
import numpy as np

# QCE ENGINE CORE (Rule-based, text-derived)
THAI_LOWER = lambda s: s  # Thai has no case; keep original for matching substrings

HEDGES = set([
    'อาจ','น่าจะ','เหมือน','คิดว่า','คง','ลอง','ถ้า','หรือเปล่า','ไหม','หรือไม่','บางที',
    'หวังว่า','เชื่อว่า',
    'maybe','might','perhaps','seems','seem','sort of','kind of','try','if','whether','hope','believe'
])

ASSERTIVES = set([
    'ฉันรู้','รู้','พร้อม','ชัดเจน','แน่ชัด','ตั้งใจ','ต้องการ','ยืนยัน','แน่นอน','ขออนุญาต',
    'i know','know','ready','clearly','definitely','certainly','intend','intention','want','confirm','agree'
])

NEG_CONFLICT = set([
    'ไม่แน่ใจ','ไม่มั่นใจ','ไม่ชัดเจน','ไม่พร้อม','ลังเล','สับสน','กลัว','กังวล','ขัดแย้ง',
    'unsure','uncertain','confused','afraid','worried','conflicted','doubt'
])

THETA_TOKENS = set([
    'นิ่ง','แก่น','เป็นหนึ่งเดียว','วงกลม','ภายใน','สภาวะ','ความหมาย','เงียบ','รู้โดยสัญชาตญาณ',
    'core','essence','circle','within','state','meaning','silence','intuitive','fundamental'
])

GAMMA_TOKENS = set([
    'พร้อม','ยืนยัน','ตกลง','เริ่ม','ทำเลย','ขอ','รับรอง','ตรง','พุ่ง','รู้ทันที','เดี๋ยวนี้',
    'ready','confirm','agree','start','do','request','ensure','direct','surge','instant','now'
])

ALPHA_TOKENS = set(['สงบ','เบา','สบาย','นิ่งๆ','ช้า','ผ่อน','calm','light','comfortable','slow','relax'])
BETA_TOKENS = set(['เพราะ','ดังนั้น','เหตุผล','วิเคราะห์','ตรรกะ','โครงสร้าง','ขั้นตอน','ข้อเท็จจริง',
    'because','therefore','reason','analyze','logic','structure','step','fact'])
DELTA_TOKENS = set(['พัก','ล้า','เหนื่อย','เจ็บ','ช้า','ฟื้น','หลับ','หยุดพัก','ผ่อน',
    'rest','tired','exhausted','pain','slow','recover','sleep','pause','relax'])

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
        result['features'] = f
    return result


# TEST CASES
if __name__ == '__main__':
    print("=" * 70)
    print("QCE ENGINE TEST - Run ใน VS Code ได้เลย")
    print("=" * 70)
    
    test_cases = [
        ["System: ทำได้ใช่ไหม?", "User: ตกลง"],
        ["System: สะดวกมั้ย?", "User: ไม่แน่ใจเท่าไหร่"],
        ["System: พร้อมไหม?", "User: ฉันรู้ชัดเจน พร้อมแล้ว"],
        ["System: OK?", "User: Maybe, if possible"],
        ["System: Ready?", "User: Definitely ready"],
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST CASE {i}")
        print(f"{'─' * 70}")
        print(f"Input: {test}")
        
        result = qce_read(test, debug=True)
        
        print(f"\n📊 METRICS:")
        print(f"  Intent:         {result['intent']}")
        print(f"  Discordance:    {result['discordance']}")
        print(f"  Consent Score:  {result['consent_score']}")
        print(f"  Wave Multiplier: {result['wave_multiplier']}")
        
        print(f"\n🌊 WAVES DETECTED:")
        print(f"  {', '.join([w.upper() for w in result['waves']])}")
        
        print(f"\n✅ DECISION:")
        print(f"  {result['status']}")
        
        print(f"\n💡 REASONS:")
        for reason in result['reasons']:
            print(f"  • {reason}")
        
        if result.get('wave_scores'):
            print(f"\n📈 WAVE SCORES (Debug):")
            for wave, score in result['wave_scores'].items():
                print(f"  {wave.upper():8} : {score:.2f}")
        
        if result.get('features'):
            print(f"\n🔍 FEATURE FLAGS (Debug):")
            for feature, flag in result['features'].items():
                print(f"  {feature:18} : {flag}")
    
    print(f"\n{'=' * 70}")
    print("Test Complete!")
    print(f"{'=' * 70}")
