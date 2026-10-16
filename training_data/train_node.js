/**
 * MITS Feedback AI Classifier — Pure Node.js
 * TF-IDF + Logistic Regression (no Python, no pip)
 * Trains on corrected dataset and saves model to feedback_classifier.json
 */
const fs   = require('fs');
const path = require('path');
const BASE = __dirname;

// ── Load CSVs ────────────────────────────────────────────────────────────────
function loadCSV(file) {
  const lines = fs.readFileSync(path.join(BASE, file), 'utf8').split('\n');
  const rows  = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    // Parse: "label","source","is_real","text"
    const m = line.match(/^"([^"]+)","([^"]+)","([^"]+)","([\s\S]*)"$/);
    if (!m) continue;
    rows.push({ label: m[1], text: m[4].replace(/""/g, '"') });
  }
  return rows;
}

const train = loadCSV('train.csv').filter(d => ['APPRECIATION','NEEDS_ATTENTION'].includes(d.label));
const val   = loadCSV('val.csv').filter(d   => ['APPRECIATION','NEEDS_ATTENTION'].includes(d.label));
const test  = loadCSV('test.csv').filter(d  => ['APPRECIATION','NEEDS_ATTENTION'].includes(d.label));

console.log(`Train: ${train.length}  Val: ${val.length}  Test: ${test.length}`);
console.log(`Train APPRECIATION:    ${train.filter(d=>d.label==='APPRECIATION').length}`);
console.log(`Train NEEDS_ATTENTION: ${train.filter(d=>d.label==='NEEDS_ATTENTION').length}\n`);

// ── Preprocessing ─────────────────────────────────────────────────────────────
function preprocess(text) {
  return text.toLowerCase()
    .replace(/[^\w\s]/g,' ')
    .replace(/\s+/g,' ')
    .trim();
}

// ── N-gram tokenizer ──────────────────────────────────────────────────────────
function getNgrams(text, n) {
  const words = preprocess(text).split(' ').filter(w => w.length > 1);
  if (n === 1) return words;
  const grams = [...words];
  for (let i = 0; i < words.length - (n-1); i++) {
    grams.push(words.slice(i, i+n).join('_'));
  }
  return grams;
}

function tokenize(text) {
  return getNgrams(text, 2); // unigrams + bigrams
}

// ── Build vocabulary from training set ────────────────────────────────────────
console.log('Building vocabulary...');
const termDocFreq = {};
const totalDocs   = train.length;

train.forEach(d => {
  const terms = new Set(tokenize(d.text));
  terms.forEach(t => {
    termDocFreq[t] = (termDocFreq[t] || 0) + 1;
  });
});

// Keep top 8000 terms by document frequency (min 2 docs)
const vocab = Object.entries(termDocFreq)
  .filter(([,v]) => v >= 2)
  .sort((a,b) => b[1]-a[1])
  .slice(0, 8000)
  .map(([t]) => t);

const termIdx = {};
vocab.forEach((t,i) => termIdx[t] = i);
console.log(`Vocabulary size: ${vocab.length}`);

// ── TF-IDF vectorize ──────────────────────────────────────────────────────────
function vectorize(text) {
  const tokens = tokenize(text);
  const tf = {};
  tokens.forEach(t => tf[t] = (tf[t]||0) + 1);
  const vec = new Float64Array(vocab.length);
  for (const [t, count] of Object.entries(tf)) {
    if (t in termIdx) {
      const tfidf = (count / tokens.length) *
        Math.log((totalDocs + 1) / ((termDocFreq[t] || 0) + 1));
      vec[termIdx[t]] = tfidf;
    }
  }
  // L2 normalize
  let norm = 0;
  vec.forEach(v => norm += v*v);
  norm = Math.sqrt(norm) || 1;
  return vec.map(v => v/norm);
}

// ── Logistic Regression ────────────────────────────────────────────────────────
// Label: 1 = NEEDS_ATTENTION, 0 = APPRECIATION
const LABELS = { 'NEEDS_ATTENTION': 1, 'APPRECIATION': 0 };

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function dotProduct(w, x) {
  let s = 0;
  for (let i = 0; i < w.length; i++) s += w[i] * x[i];
  return s;
}

// Train with SGD
console.log('\nVectorizing training data...');
const X_train = train.map(d => vectorize(d.text));
const y_train = train.map(d => LABELS[d.label]);

const W = new Float64Array(vocab.length); // weights
let bias = 0;
const LR   = 0.5;
const EPOCHS = 30;
const LAMBDA = 0.001; // L2 regularization

console.log(`Training ${EPOCHS} epochs...`);
for (let epoch = 0; epoch < EPOCHS; epoch++) {
  let loss = 0;
  // Shuffle
  const indices = Array.from({length: X_train.length}, (_,i) => i)
    .sort(() => Math.random() - 0.5);

  for (const i of indices) {
    const x = X_train[i];
    const y = y_train[i];
    const z = dotProduct(W, x) + bias;
    const pred = sigmoid(z);
    const err  = pred - y;
    loss += -(y * Math.log(pred + 1e-9) + (1-y) * Math.log(1-pred + 1e-9));

    // Gradient update
    for (let j = 0; j < W.length; j++) {
      W[j] = W[j] * (1 - LR * LAMBDA) - LR * err * x[j];
    }
    bias -= LR * err;
  }

  if ((epoch + 1) % 5 === 0) {
    process.stdout.write(`  Epoch ${epoch+1}/${EPOCHS}  loss=${(loss/X_train.length).toFixed(4)}\r`);
  }
}
console.log('\nTraining complete.\n');

// ── Evaluate ──────────────────────────────────────────────────────────────────
function predict(text, threshold = 0.45) {
  const x    = vectorize(text);
  const prob = sigmoid(dotProduct(W, x) + bias);
  if (prob >= threshold) return { label: 'NEEDS_ATTENTION', prob };
  return { label: 'APPRECIATION', prob: 1 - prob };
}

function evaluate(dataset, name, threshold = 0.45) {
  let tp=0, fp=0, tn=0, fn=0;
  dataset.forEach(d => {
    const pred = predict(d.text, threshold).label;
    const actual = d.label;
    if (pred==='NEEDS_ATTENTION' && actual==='NEEDS_ATTENTION') tp++;
    else if (pred==='NEEDS_ATTENTION' && actual==='APPRECIATION')  fp++;
    else if (pred==='APPRECIATION'   && actual==='APPRECIATION')   tn++;
    else if (pred==='APPRECIATION'   && actual==='NEEDS_ATTENTION')fn++;
  });
  const prec = tp/(tp+fp||1);
  const rec  = tp/(tp+fn||1);
  const f1   = 2*prec*rec/(prec+rec||1);
  const acc  = (tp+tn)/dataset.length;
  console.log(`── ${name} ────────────────────────`);
  console.log(`  Accuracy:  ${(acc*100).toFixed(1)}%`);
  console.log(`  Precision: ${(prec*100).toFixed(1)}% (NEEDS_ATTENTION)`);
  console.log(`  Recall:    ${(rec*100).toFixed(1)}% (NEEDS_ATTENTION)`);
  console.log(`  F1 Score:  ${(f1*100).toFixed(1)}%`);
  console.log(`  TP:${tp} FP:${fp} TN:${tn} FN:${fn}`);
  if (fn > 0) {
    console.log(`  FALSE NEGATIVES (NEEDS_ATTENTION → predicted APPRECIATION):`);
    dataset.filter(d => d.label==='NEEDS_ATTENTION' && predict(d.text,threshold).label==='APPRECIATION')
      .slice(0,5).forEach(d => console.log(`    - ${d.text.substring(0,80)}`));
  }
  console.log();
  return { tp,fp,tn,fn,acc,prec,rec,f1 };
}

// Find optimal threshold on validation set
let bestThreshold = 0.45, bestF1 = 0;
for (let t = 0.30; t <= 0.70; t += 0.05) {
  let tp=0,fp=0,tn=0,fn=0;
  val.forEach(d => {
    const pred = predict(d.text, t).label;
    if (pred==='NEEDS_ATTENTION' && d.label==='NEEDS_ATTENTION') tp++;
    else if (pred==='NEEDS_ATTENTION' && d.label==='APPRECIATION') fp++;
    else if (pred==='APPRECIATION' && d.label==='APPRECIATION') tn++;
    else fn++;
  });
  const prec = tp/(tp+fp||1), rec = tp/(tp+fn||1);
  const f1   = 2*prec*rec/(prec+rec||1);
  if (f1 > bestF1) { bestF1=f1; bestThreshold=parseFloat(t.toFixed(2)); }
}
console.log(`Best threshold (val): ${bestThreshold}  F1: ${(bestF1*100).toFixed(1)}%\n`);

evaluate(val,  'VALIDATION SET', bestThreshold);
evaluate(test, 'TEST SET (held-out real)', bestThreshold);

// ── Save model ────────────────────────────────────────────────────────────────
const model = {
  version:    'v1.0-nodejs',
  trained_at: new Date().toISOString(),
  threshold:  bestThreshold,
  vocab,
  termDocFreq,
  totalDocs,
  weights:    Array.from(W),
  bias,
  train_size: train.length,
  val_size:   val.length,
  test_size:  test.length,
};
const modelPath = path.join(BASE, 'feedback_classifier.json');
fs.writeFileSync(modelPath, JSON.stringify(model));
console.log(`Model saved: training_data/feedback_classifier.json`);
console.log(`Size: ${(fs.statSync(modelPath).size/1024).toFixed(0)} KB`);
