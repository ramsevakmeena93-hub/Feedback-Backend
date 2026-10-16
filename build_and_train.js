/**
 * BUILD CORRECTED DATASET + SAVE FOR PYTHON TRAINING
 * Applies all manual corrections, adds synthetic data,
 * splits train/val/test, saves CSV files for Python trainer.
 */
const mongoose = require('mongoose');
const fs = require('fs');
const path = require('path');

const MONGO_URI = 'mongodb+srv://studentmeena088_db_user:9jKWHcKRc8b2g9LZ@cluster0.peoobft.mongodb.net/faculty_feedback?retryWrites=true&w=majority&appName=Cluster0';
const OUT = path.join(__dirname, 'training_data');

// ─── MANUAL CORRECTIONS from user review ────────────────────────────────────
// These comment IDs (1-based index from show_comments output) are in APPRECIATION
// but should be NEEDS_ATTENTION
const MOVE_TO_ATTENTION = new Set([
  17,18,19,33,49,59,62,97,122,123,151,167,168,170,171,172,174,183,203,
  278,282,283,291,296,297,311,312,314,321,326,331,333,338,340,343,344,
  348,349,352,360,362,368,372,376,384,392,393,397,401,407,408,409,412,
  418,420,423,424,428,430,436,437,439,440,441,443,452,453,454,455,456,
  457,458,459,460,468,471,472,473,476,480,484,485,486,487,488,490,492,
  497,498,502,506,514,515,517,518,519,521,522,523,524,525,528,529,530,
  531,532,533,537,539,540,554,560,562,568,584,585,586,587,588,589,594,
  598,599,601,602,603,605,609,616,618
]);

// Comments that are clearly junk/no feedback
const JUNK_IDS = new Set([1, 2]); // jbjkkjb, hvhjv

// Comments misclassified as NEEDS_ATTENTION but are actually APPRECIATION
const MOVE_TO_APPRECIATION = new Set([40, 46, 97, 123, 124, 125, 126, 375, 543, 552, 553, 566]);

async function build() {
  await mongoose.connect(MONGO_URI);
  console.log('MongoDB connected\n');

  const Report = mongoose.model('FacultyReport', new mongoose.Schema({}, { strict: false }), 'facultyreports');
  const reports = await Report.find({}).lean();

  // Collect all unique comments with original label + sequential index
  const seen = new Set();
  const allComments = [];
  let idx = 0;

  for (const r of reports) {
    const add = (arr, origLabel) => {
      for (const c of (arr || [])) {
        const key = c?.trim().toLowerCase();
        if (!key || key.length < 3 || seen.has(key)) continue;
        seen.add(key);
        idx++;
        allComments.push({ idx, text: c.trim(), orig_label: origLabel, faculty: r.facultyName || '' });
      }
    };
    add(r.appreciation,             'APPRECIATION');
    add(r.commentsNeedingAttention, 'NEEDS_ATTENTION');
  }

  console.log(`Total unique comments from DB: ${allComments.length}`);

  // Apply corrections
  let corrected = 0;
  let junkCount = 0;
  const dataset = [];

  for (const item of allComments) {
    let label = item.orig_label;

    if (JUNK_IDS.has(item.idx)) {
      junkCount++;
      continue; // skip junk
    }
    if (MOVE_TO_ATTENTION.has(item.idx)) {
      label = 'NEEDS_ATTENTION';
      if (item.orig_label !== 'NEEDS_ATTENTION') corrected++;
    }
    if (MOVE_TO_APPRECIATION.has(item.idx)) {
      label = 'APPRECIATION';
      if (item.orig_label !== 'APPRECIATION') corrected++;
    }

    dataset.push({
      text:   item.text,
      label,
      source: 'real',
      is_real: true,
      faculty: item.faculty,
      orig_label: item.orig_label,
      corrected: label !== item.orig_label,
    });
  }

  const appCount = dataset.filter(d => d.label === 'APPRECIATION').length;
  const attCount = dataset.filter(d => d.label === 'NEEDS_ATTENTION').length;

  console.log(`Corrections applied: ${corrected}`);
  console.log(`Junk removed:        ${junkCount}`);
  console.log(`APPRECIATION:        ${appCount}`);
  console.log(`NEEDS_ATTENTION:     ${attCount}`);
  console.log(`Total usable:        ${dataset.length}\n`);

  // ── Load synthetic 1000 NEEDS_ATTENTION ─────────────────────────────────
  const synRaw = fs.readFileSync(path.join(__dirname, 'training_data', 'synthetic_comments.csv'), 'utf8');
  const synLines = synRaw.split('\n').slice(1); // skip header
  const synDataset = [];
  for (const line of synLines) {
    if (!line.trim()) continue;
    const match = line.match(/^[^,]+,"((?:[^"\\]|\\.|"")*)"/);
    if (!match) continue;
    const text = match[1].replace(/""/g, '"').trim();
    if (text.length > 5) {
      synDataset.push({ text, label: 'NEEDS_ATTENTION', source: 'synthetic', is_real: false });
    }
  }
  console.log(`Synthetic NEEDS_ATTENTION loaded: ${synDataset.length}`);

  // ── Split ──────────────────────────────────────────────────────────────
  // Shuffle real data deterministically
  const usable = dataset.filter(d => ['APPRECIATION','NEEDS_ATTENTION'].includes(d.label));
  usable.sort((a, b) => simpleHash(a.text) - simpleHash(b.text));

  const n     = usable.length;
  const testN = Math.floor(n * 0.10);
  const valN  = Math.floor(n * 0.10);

  const testSet  = usable.slice(0, testN);
  const valSet   = usable.slice(testN, testN + valN);
  const trainReal= usable.slice(testN + valN);
  const trainSet = [...trainReal, ...synDataset];

  console.log(`Train: ${trainSet.length} (${trainReal.length} real + ${synDataset.length} synthetic)`);
  console.log(`Val:   ${valSet.length}`);
  console.log(`Test:  ${testSet.length} (held-out real only)\n`);

  // ── Save CSVs ──────────────────────────────────────────────────────────
  if (!fs.existsSync(OUT)) fs.mkdirSync(OUT, { recursive: true });

  const toCSV = (rows) => {
    const hdr = 'label,source,is_real,text\n';
    return hdr + rows.map(d =>
      `"${d.label}","${d.source||'real'}","${d.is_real!==false}","${(d.text||'').replace(/"/g,'""')}"`
    ).join('\n');
  };

  fs.writeFileSync(path.join(OUT, 'train.csv'), toCSV(trainSet));
  fs.writeFileSync(path.join(OUT, 'val.csv'),   toCSV(valSet));
  fs.writeFileSync(path.join(OUT, 'test.csv'),  toCSV(testSet));
  fs.writeFileSync(path.join(OUT, 'full_real_corrected.json'), JSON.stringify(dataset, null, 2));

  console.log('Saved: train.csv, val.csv, test.csv, full_real_corrected.json');
  console.log('\nLabel distribution in train:');
  const tApp = trainSet.filter(d => d.label==='APPRECIATION').length;
  const tAtt = trainSet.filter(d => d.label==='NEEDS_ATTENTION').length;
  console.log(`  APPRECIATION:    ${tApp}`);
  console.log(`  NEEDS_ATTENTION: ${tAtt}`);

  process.exit(0);
}

function simpleHash(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return h >>> 0;
}

build().catch(e => { console.error(e.message); process.exit(1); });
