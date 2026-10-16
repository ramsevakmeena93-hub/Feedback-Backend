/**
 * AI Comment Analyzer — TF-IDF + Logistic Regression (trained model)
 * Trained on corrected MITS feedback dataset (real + synthetic)
 * Falls back to rule-based if model not found.
 */
const fs   = require('fs');
const path = require('path');

// ── Load trained model ────────────────────────────────────────────────────────
let MODEL = null;
const MODEL_PATH = path.join(__dirname, '../training_data/feedback_classifier.json');

function loadModel() {
  if (MODEL) return MODEL;
  try {
    MODEL = JSON.parse(fs.readFileSync(MODEL_PATH, 'utf8'));
    console.log(`[AI] Model loaded: ${MODEL.version} (${MODEL.vocab.length} features, threshold=${MODEL.threshold})`);
  } catch (e) {
    console.warn('[AI] Model not found, using rule-based fallback:', e.message);
  }
  return MODEL;
}

// ── Preprocessing ─────────────────────────────────────────────────────────────
function preprocess(text) {
  return (text||'').toLowerCase().replace(/[^\w\s]/g,' ').replace(/\s+/g,' ').trim();
}

function tokenize(text) {
  const words = preprocess(text).split(' ').filter(w => w.length > 1);
  const grams = [...words];
  for (let i = 0; i < words.length - 1; i++) {
    grams.push(words[i] + '_' + words[i+1]);
  }
  return grams;
}

function vectorize(text, model) {
  const tokens = tokenize(text);
  const tf = {};
  tokens.forEach(t => tf[t] = (tf[t]||0)+1);
  const vec = new Float64Array(model.vocab.length);
  const termIdx = model._termIdx || (model._termIdx = (() => {
    const idx = {};
    model.vocab.forEach((t,i) => idx[t]=i);
    return idx;
  })());

  for (const [t, count] of Object.entries(tf)) {
    if (t in termIdx) {
      const idf = Math.log((model.totalDocs+1) / ((model.termDocFreq[t]||0)+1));
      vec[termIdx[t]] = (count / tokens.length) * idf;
    }
  }
  let norm = 0;
  vec.forEach(v => norm += v*v);
  norm = Math.sqrt(norm) || 1;
  return vec.map(v => v/norm);
}

function sigmoid(z) { return 1/(1+Math.exp(-z)); }

function modelPredict(text) {
  const m = loadModel();
  if (!m) return null;
  const x   = vectorize(text, m);
  const W   = m.weights;
  let dot   = m.bias;
  for (let i = 0; i < W.length; i++) dot += W[i] * x[i];
  const prob = sigmoid(dot); // P(NEEDS_ATTENTION)
  return { prob, label: prob >= m.threshold ? 'NEEDS_ATTENTION' : 'APPRECIATION' };
}

// ── Rule-based fallback for clear-cut cases ───────────────────────────────────
const CLEAR_NEGATIVE = [
  /too fast/i,/very fast/i,/speaks fast/i,/jaldi/i,/slow down/i,
  /not clear/i,/unclear/i,/samajh nahi/i,/samajh nhi/i,/kuch nahi samjha/i,
  /kuchh nahin samajh/i,/smjh nhi/i,/kuch smjh/i,/teaching skill khrab/i,
  /notes nahi/i,/notes are never/i,/notes are not shared/i,/no notes/i,
  /please share notes/i,/please upload/i,/provide notes/i,/share notes/i,
  /notes de dia kro/i,/ppts on time/i,/notes on time/i,
  /never seen/i,/have never seen/i,/we have never seen/i,
  /irregular/i,/absent/i,/class miss/i,/class nahi/i,/late aate/i,
  /need to improve/i,/needs improvement/i,/should improve/i,/must improve/i,
  /hope he improve/i,/hope she improve/i,/please improve/i,
  /average/i,/bad/i,/poor/i,/worst/i,/terrible/i,/boring/i,/not engaging/i,
  /not interactive/i,/not satisfied/i,/disappointing/i,
  /bakwas/i,/bekar/i,/faltu/i,/khrab/i,/theek nahi/i,/accha nahi/i,
  /ratta maar/i,/please change/i,/change faculty/i,/change the faculty/i,
  /partial with marks/i,/out of syllabus/i,/marks not justified/i,
  /not approachable/i,/no interaction/i,/does not help/i,/not available/i,
  /copy paste/i,/lab not interactive/i,/kits kam/i,/computers not working/i,
  /internet slow/i,/software not/i,/equipment not/i,
];

const CLEAR_POSITIVE = [
  /excellent/i,/outstanding/i,/superb/i,/brilliant/i,/best teacher/i,
  /great teacher/i,/good teacher/i,/amazing teacher/i,/wonderful/i,
  /very good/i,/really good/i,/very helpful/i,/very supportive/i,
  /keep it up/i,/well done/i,/thank you/i,/thanks/i,/grateful/i,
  /fully satisfied/i,/completely satisfied/i,/very satisfied/i,
  /no remark needed/i,/no complaints/i,/everything is good/i,
  /everything is fine/i,/all is good/i,/all is well/i,
  /best faculty/i,/best mam/i,/best sir/i,/lovely faculty/i,
  /bahut accha/i,/bhot acha/i,/zabardast/i,/sahi hai/i,
  /my inspiration/i,/dedicated/i,/hardworking/i,
  /hats off/i,/cooperative/i,/kind and humble/i,/friendly/i,
  /i really like/i,/i love the way/i,/personally like/i,
  /pls teach again/i,/please assign again/i,/please assign them/i,
];

function ruleBasedClassify(text) {
  const isNeg = CLEAR_NEGATIVE.some(p => p.test(text));
  if (isNeg) return { label: 'NEEDS_ATTENTION', conf: 0.92, method: 'rule' };
  const isPos = CLEAR_POSITIVE.some(p => p.test(text));
  if (isPos) return { label: 'APPRECIATION', conf: 0.90, method: 'rule' };
  return null;
}

// ── Main classifier ───────────────────────────────────────────────────────────
function classifyComment(text) {
  const t = (text||'').trim();
  if (!t || t.length < 3) return { label: 'APPRECIATION', conf: 0.5, method: 'skip' };

  // 1. Rule-based for clear-cut cases (fast + high precision)
  const ruleResult = ruleBasedClassify(t);
  if (ruleResult && ruleResult.conf >= 0.90) return ruleResult;

  // 2. Trained model
  const mlResult = modelPredict(t);
  if (mlResult) {
    const conf = mlResult.label === 'NEEDS_ATTENTION' ? mlResult.prob : 1 - mlResult.prob;
    return { label: mlResult.label, conf, method: 'model' };
  }

  // 3. Rule-based fallback
  return ruleResult || { label: 'APPRECIATION', conf: 0.5, method: 'fallback' };
}

// ── Category patterns ─────────────────────────────────────────────────────────
const CATEGORY_PATTERNS = {
  Speed:        /\b(fast|slow|speed|pace|jaldi|rush|bolne ki speed)\b/i,
  Clarity:      /\b(unclear|confusing|understand|explain|clear|samajh|concept)\b/i,
  Materials:    /\b(notes|material|slide|pdf|book|share|provide|upload)\b/i,
  Interaction:  /\b(available|interaction|doubt|question|help|approachable|cooperative|listen)\b/i,
  Regularity:   /\b(irregular|regular|punctual|late|miss|skip|absent|cancel|class nahi)\b/i,
};

// ── Main API ──────────────────────────────────────────────────────────────────
async function analyzeCommentsWithAI(rawComments) {
  loadModel(); // warm up

  const result = {
    appreciation: [],
    commentsNeedingAttention: [],
    commentCategories: { Speed:[], Clarity:[], Materials:[], Interaction:[], Regularity:[], General:[] }
  };

  if (!rawComments || rawComments.length === 0) return result;

  const SPLIT_REGEX = /\b(?:but|however|though|although|yet|still|lekin|par|magar|except)\b/i;

  function addAttention(text) {
    result.commentsNeedingAttention.push(text);
    let categorized = false;
    for (const [cat, regex] of Object.entries(CATEGORY_PATTERNS)) {
      if (regex.test(text)) { result.commentCategories[cat].push(text); categorized = true; }
    }
    if (!categorized) result.commentCategories.General.push(text);
  }

  for (const comment of rawComments) {
    if (!comment || typeof comment !== 'string') continue;
    const trimmed = comment.trim();
    if (trimmed.length < 3) continue;

    // Split on conjunctions — handle mixed feedback
    const parts = trimmed.split(SPLIT_REGEX).map(p => p.trim()).filter(p => p.length > 2);

    if (parts.length > 1) {
      // If any part is NEEDS_ATTENTION, whole comment goes to attention
      const hasNeg = parts.some(p => classifyComment(p).label === 'NEEDS_ATTENTION');
      if (hasNeg) addAttention(trimmed);
      else result.appreciation.push(trimmed);
      continue;
    }

    const { label } = classifyComment(trimmed);
    if (label === 'NEEDS_ATTENTION') addAttention(trimmed);
    else result.appreciation.push(trimmed);
  }

  console.log(`[AI] ${rawComments.length} comments → ${result.appreciation.length} appreciation, ${result.commentsNeedingAttention.length} needs attention`);
  return result;
}

async function testGeminiConnection() {
  loadModel();
  const testComments = [
    'Very good teaching',
    'needs to improve speed',
    'excellent mam',
    'Good teaching but too fast',
    'not sharing notes',
    'boring class',
    'samajh nahi aata',
    'best teacher',
  ];
  const result = await analyzeCommentsWithAI(testComments);
  return {
    ok: true,
    response: `Model v1.0-nodejs — ${result.appreciation.length} appreciation, ${result.commentsNeedingAttention.length} attention`,
    engine: 'TF-IDF + Logistic Regression (trained on MITS data)',
    modelVersion: MODEL ? MODEL.version : 'rule-based-fallback',
    test: result,
  };
}

module.exports = { analyzeCommentsWithAI, testGeminiConnection };
