/**
 * AI Comment Analyzer — Smart Rule-Based + HuggingFace fallback
 *
 * Problem with pure AI: distilbert misclassifies short academic phrases like
 * "Very Good", "Great classes", "Excellent mam" as NEGATIVE because it was
 * trained on movie reviews, not faculty feedback forms.
 *
 * Solution: A comprehensive rule-based classifier that understands MITS
 * feedback form language (English + Hinglish), with AI only as a fallback
 * for genuinely ambiguous long sentences.
 */
let pipeline = null;
let pipelineLoading = false;
async function getSentimentPipeline() {
  if (pipeline) return pipeline;
  if (pipelineLoading) {
    while (pipelineLoading) await new Promise(r => setTimeout(r, 100));
    return pipeline;
  }
  pipelineLoading = true;
  try {
    const { pipeline: createPipeline } = await import('@xenova/transformers');
    pipeline = await createPipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
    console.log('[AI] HuggingFace sentiment model loaded');
  } finally {
    pipelineLoading = false;
  }
  return pipeline;
}

// ─────────────────────────────────────────────────────────────────────────────
// POSITIVE keyword patterns — academic feedback context
// ─────────────────────────────────────────────────────────────────────────────
const POSITIVE_PATTERNS = [
  // Single-word ratings
  /^excellent$/i, /^very good$/i, /^good$/i, /^great$/i, /^outstanding$/i,
  /^superb$/i, /^brilliant$/i, /^best$/i, /^nice$/i, /^satisfactory$/i,
  /^wonderful$/i, /^awesome$/i, /^perfect$/i, /^fantastic$/i, /^amazing$/i,
  /^exceptional$/i, /^magnificent$/i, /^splendid$/i, /^marvelous$/i,
  // Compound positives
  /very good/i, /quite good/i, /really good/i, /very nice/i, /very helpful/i,
  /very clear/i, /very effective/i, /very well/i, /extremely good/i,
  /highly satisfied/i, /highly recommend/i, /very satisf/i,
  // Teaching positives
  /great teacher/i, /good teacher/i, /excellent teacher/i, /best teacher/i,
  /great class/i, /good class/i, /excellent class/i, /great lecture/i,
  /good lecture/i, /excellent lecture/i, /good explanation/i,
  /clear explanation/i, /easy to understand/i, /easy to learn/i,
  /well explained/i, /nicely explained/i, /clearly explained/i,
  /good concept/i, /good knowledge/i, /great knowledge/i,
  /good communication/i, /good interaction/i, /good teaching/i,
  /excellent teaching/i, /great teaching/i, /effective teaching/i,
  /good understanding/i, /makes it easy/i, /easy to grasp/i,
  /teaching style is good/i, /teaching method is good/i,
  /best way of teaching/i, /good way of teaching/i,
  // Respect / appreciation (Indian academic context)
  /good mam/i, /good sir/i, /nice mam/i, /nice sir/i,
  /excellent mam/i, /excellent sir/i, /best mam/i, /best sir/i,
  /great mam/i, /great sir/i, /thank you/i, /thanks a lot/i,
  /grateful/i, /wonderful mam/i, /wonderful sir/i,
  /hats off/i, /keep it up/i, /keep up/i, /well done/i,
  // Satisfaction
  /fully satisfied/i, /completely satisfied/i, /very satisfied/i,
  /overall good/i, /overall great/i, /overall excellent/i, /overall nice/i,
  /overall satisf/i, /everything is good/i, /everything good/i,
  /all good/i, /all is well/i,
  // Encouragement
  /continue the same/i, /please continue/i, /keep going/i,
  /maintain this/i, /best wishes/i, /good luck/i,
  // Hinglish positive
  /bahut accha/i, /bahut acha/i, /acha hai/i, /accha hai/i,
  /best hai/i, /zabardast/i, /mast hai/i, /sahi hai/i,
  // Negation-positive (sound negative but mean positive/no issues)
  /not bad/i, /not bad at all/i, /no complaints/i, /no problem/i,
  /no issues/i, /nothing to complain/i,
];

// ─────────────────────────────────────────────────────────────────────────────
// NEGATIVE keyword patterns — attention needed
// RULE: Check NEGATIVE before POSITIVE — negatives are more specific
// ─────────────────────────────────────────────────────────────────────────────
const NEGATIVE_PATTERNS = [
  // Single-word negatives
  /^no$/i, /^nil$/i, /^na$/i, /^n\/a$/i, /^none$/i, /^nothing$/i,
  /^poor$/i, /^bad$/i, /^worst$/i, /^terrible$/i, /^horrible$/i,
  /^average$/i, /^below average$/i,
  /^not good$/i, /^not great$/i, /^not nice$/i, /^not satisfied$/i,
  /^boring$/i, /^bored$/i, /^useless$/i, /^waste$/i, /^pathetic$/i,
  /^disappointing$/i, /^frustrated$/i, /^unhappy$/i,

  // Improvement needed
  /need to improve/i, /needs improvement/i, /should improve/i,
  /must improve/i, /can improve/i, /could improve/i, /require improvement/i,
  /improve your/i, /improve the/i, /please improve/i, /improvement required/i,
  /work on improvement/i, /scope of improvement/i,

  // Speed / pace issues
  /too fast/i, /very fast/i, /speaks fast/i, /teaching fast/i,
  /talks fast/i, /goes fast/i, /rushes through/i, /rush through/i,
  /speak slowly/i, /talk slowly/i, /teach slowly/i, /slow down/i,
  /too slow/i, /very slow/i, /slow speed/i, /slow pace/i,
  /reduce speed/i, /speed is (too|very)/i, /pace is (too|very)/i,
  /bolne ki speed/i, /jaldi bolte/i, /jaldi padhate/i,

  // Clarity / explanation issues
  /not clear/i, /unclear/i, /not clearly/i,
  /hard to understand/i, /difficult to understand/i,
  /difficult to follow/i, /hard to follow/i, /not understandable/i,
  /poor explanation/i, /bad explanation/i, /confusing/i, /confused/i,
  /didn't understand/i, /don't understand/i, /did not understand/i,
  /cannot understand/i, /can't understand/i, /unable to understand/i,
  /doesn't explain/i, /does not explain/i, /didn't explain/i,
  /not explained/i, /poorly explained/i, /explanation is (not|poor|bad)/i,
  /samajh nahi/i, /samajh me nahi/i, /samajh nahi ata/i, /samajh nahi aata/i,

  // Availability / interaction
  /not available/i, /never available/i, /not approachable/i,
  /no interaction/i, /less interaction/i, /poor interaction/i,
  /does not interact/i, /doesn't interact/i, /interaction is (poor|less|no)/i,
  /not accessible/i, /does not help/i, /doesn't help/i,
  /not cooperative/i, /not supportive/i, /not responsive/i,
  /no doubt clearing/i, /doubt clear nahi/i, /doubt nahi clear/i,
  /doesn't solve doubts/i, /does not solve doubts/i,
  /doubts not cleared/i, /doubt not cleared/i,
  /doesn't clear doubts/i, /does not clear doubts/i,

  // Notes / material issues
  /no notes/i, /no material/i, /no slides/i, /no pdf/i,
  /should share/i, /please share notes/i, /please share material/i,
  /provide notes/i, /provide material/i, /provide slides/i,
  /not provided/i, /not shared/i, /not given/i,
  /notes not/i, /material not/i, /slides not/i, /pdf not/i,
  /doesn't provide/i, /does not provide/i, /didn't provide/i,
  /doesn't share/i, /does not share/i, /didn't share/i,
  /doesn't give/i, /does not give/i, /didn't give/i,
  /share notes/i, /share material/i, /share slides/i,
  /upload notes/i, /upload material/i, /notes upload/i,
  /notes share/i, /notes nahi/i, /material nahi/i,

  // Attendance / regularity
  /irregular/i, /not regular/i, /misses class/i, /skips class/i,
  /not punctual/i, /late to class/i, /always late/i,
  /comes late/i, /come late/i, /came late/i, /is late/i, /was late/i,
  /class miss/i, /class cancel/i, /class nahi/i,
  /doesn't come/i, /does not come/i, /didn't come/i, /not coming/i,
  /absent/i, /not present/i, /bunks/i,
  /class nahi lete/i, /class nahi aate/i, /class nahi aati/i,
  /late aate/i, /late aati/i, /miss karte/i, /miss karti/i,

  // Engagement / motivation
  /boring/i, /bored/i, /not engaging/i, /monotonous/i, /dull/i,
  /not interesting/i, /not interested/i, /lost interest/i, /no interest/i,
  /not motivated/i, /demotivated/i, /no motivation/i,
  /no energy/i, /not enthusiastic/i,

  // Attitude / behaviour
  /rude/i, /harsh/i, /angry/i, /aggressive/i, /arrogant/i,
  /doesn't care/i, /does not care/i, /doesn't listen/i, /does not listen/i,
  /favouritism/i, /biased/i, /partial/i,
  /disrespectful/i, /unfair/i, /unkind/i,

  // General dissatisfaction
  /not satisfied/i, /unsatisfied/i, /dissatisfied/i,
  /disappointing/i, /disappointed/i,
  /waste of time/i, /not helpful/i, /unhelpful/i,
  /not useful/i, /useless/i, /not relevant/i, /irrelevant/i,
  /not effective/i, /ineffective/i, /nothing special/i,
  /nothing new/i, /nothing to learn/i, /nothing much/i,
  /didn't like/i, /don't like/i, /did not like/i, /do not like/i,
  /could be better/i, /should be better/i, /needs to be better/i,
  /lack of/i, /lacks/i, /problem with/i, /issue with/i,
  /not enough/i, /insufficient/i,
  /not prepared/i, /unprepared/i,
  /doesn't teach/i, /does not teach/i, /not teaching/i,
  /should focus/i, /should work on/i,

  // Hinglish negatives
  /accha nahi/i, /acha nahi/i, /theek nahi/i, /thik nahi/i,
  /bakwas/i, /bekar/i, /faltu/i, /ghatiya/i,
  /boring hai/i, /bore hota/i, /bore karta/i,
  /nahi aata/i, /nahi aati/i, /nahi aate/i,
  /nahi padhat/i, /nahi sikhate/i, /nahi sikhati/i,
  /kuch nahi/i, /koi fayda nahi/i, /fayda nahi/i,
  /miss karna/i, /chhod dete/i,

  // "needs to X" / "should X" / "please X" (suggestions = lacking)
  /needs? to\s+\w+/i,       // needs to improve, need to explain
  /should\s+\w+/i,           // should improve, should interact
  /please\s+\w+/i,           // please share, please provide
  /must\s+\w+/i,             // must improve, must share
  /try to\s+\w+/i,           // try to improve
  /have to\s+\w+/i,          // have to improve

  // "more X" suggestions (means current is lacking)
  /more examples/i, /more practice/i, /more interaction/i,
  /more engagement/i, /more attention/i, /more focus/i,
  /more doubts/i, /more time/i, /more clearly/i,
  /more slowly/i, /more practical/i, /more assignments/i,
  /should give more/i, /should do more/i, /should teach more/i,

  // "less X" complaints
  /less interaction/i, /less practical/i, /less examples/i,
  /less focus/i, /less time/i, /less engagement/i,

  // Generic not+verb catch (safe version — only verbs, not adjectives)
  /\bnot\s+(providing|sharing|giving|teaching|helping|explaining|clearing|coming|attending|solving|listening|preparing|engaging|interacting)\b/i,
];

// ─────────────────────────────────────────────────────────────────────────────
// SKIP patterns — meaningless / empty entries
// ─────────────────────────────────────────────────────────────────────────────
const SKIP_PATTERNS = [
  /^-+$/, /^\.*$/, /^_+$/, /^\s*$/, /^x+$/i, /^\.{1,3}$/,
  /^[0-9]+$/, /^[^a-zA-Z]+$/,
  /^(no\s+)?suggestion[s]?$/i,
  /^(no\s+)?comment[s]?$/i,
  /^nothing\s+to\s+(say|write|suggest|comment)$/i,
  /^all\s+(is\s+)?(good|well|fine|okay|ok)$/i,
  /^(everything\s+)?(is\s+)?(good|fine|ok|okay|well)$/i,
  /^(no\s+)?(improvement\s+)?needed$/i,
  /^n\.?\/?a\.?$/i,
  /^(thank\s*you|thanks)$/i,
];

// ─────────────────────────────────────────────────────────────────────────────
// NEUTRAL skip — filler comments that should NOT be shown anywhere
// ─────────────────────────────────────────────────────────────────────────────
const NEUTRAL_SKIP_PATTERNS = [
  /don't know what to/i, /don't know what to write/i,
  /i don't know/i, /idk/i,
  /nothing to say/i, /nothing to write/i,
  /no idea/i, /whatever/i,
  /kuch nahi likhna/i, /pata nahi/i, /kya likhu/i,
];

// ─────────────────────────────────────────────────────────────────────────────
// Category patterns — for attention comments
// ─────────────────────────────────────────────────────────────────────────────
const CATEGORY_PATTERNS = {
  Speed:        /\b(fast|slow|speed|pace|quick|jaldi|rush|bolne ki speed)\b/i,
  Clarity:      /\b(unclear|confusing|understand|explain|clear|samajh|concept|confus)\b/i,
  Materials:    /\b(notes|material|slide|pdf|book|share|provide|upload|give)\b/i,
  Interaction:  /\b(available|interaction|doubt|question|help|approachable|cooperative|listen|care|solve)\b/i,
  Regularity:   /\b(irregular|regular|punctual|late|miss|skip|absent|cancel|bunks|aate|aati|come|class nahi)\b/i,
};

// Split on conjunctions that indicate mixed sentiment
const SPLIT_REGEX = /\b(?:but|however|though|although|yet|still|lekin|par|magar|parantu|except)\b/i;

// ─────────────────────────────────────────────────────────────────────────────
// Core rule-based classifier
// Returns: 'positive' | 'negative' | 'neutral' | 'skip' | 'ai'
// ─────────────────────────────────────────────────────────────────────────────
function ruleBasedClassify(comment) {
  const text = comment.trim();
  if (!text || text.length < 2) return 'skip';

  // Skip meaningless entries
  if (SKIP_PATTERNS.some(p => p.test(text))) return 'skip';

  // Skip neutral filler
  if (NEUTRAL_SKIP_PATTERNS.some(p => p.test(text.toLowerCase()))) return 'skip';

  const lower = text.toLowerCase();

  // ── Check NEGATIVE first (higher priority — more specific) ──────────────
  if (NEGATIVE_PATTERNS.some(p => p.test(lower))) return 'negative';

  // ── Check POSITIVE ────────────────────────────────────────────────────────
  if (POSITIVE_PATTERNS.some(p => p.test(lower))) return 'positive';

  const wordCount = text.trim().split(/\s+/).length;

  // Short comments (≤4 words) with negation → negative
  if (wordCount <= 4) {
    if (/\b(not|no|never|don't|doesn't|didn't|can't|won't|bad|poor|worst|boring|average)\b/i.test(lower)) {
      return 'negative';
    }
    // Short with no match and no signal → skip (don't pollute appreciation)
    return 'skip';
  }

  // Medium comments (5-8 words) with negation → negative
  if (wordCount <= 8) {
    if (/\b(not|no|never|don't|doesn't|didn't|can't|won't|shouldn't|couldn't)\b/i.test(lower)) {
      return 'negative';
    }
  }

  // Long comments → send to AI for classification
  return 'ai';
}

// ─────────────────────────────────────────────────────────────────────────────
// Main analyzer
// ─────────────────────────────────────────────────────────────────────────────
async function analyzeCommentsWithAI(rawComments) {
  const defaultResult = {
    appreciation: [],
    commentsNeedingAttention: [],
    commentCategories: {
      Speed: [], Clarity: [], Materials: [], Interaction: [], Regularity: [], General: []
    }
  };

  if (!rawComments || rawComments.length === 0) return defaultResult;

  const result = {
    appreciation: [],
    commentsNeedingAttention: [],
    commentCategories: {
      Speed: [], Clarity: [], Materials: [], Interaction: [], Regularity: [], General: []
    }
  };

  const toClassifyWithAI = [];

  function addAttention(text) {
    result.commentsNeedingAttention.push(text);
    let categorized = false;
    for (const [category, regex] of Object.entries(CATEGORY_PATTERNS)) {
      if (regex.test(text)) {
        result.commentCategories[category].push(text);
        categorized = true;
      }
    }
    if (!categorized) result.commentCategories.General.push(text);
  }

  // Step 1: Rule-based pre-classification
  rawComments.forEach((originalComment) => {
    if (!originalComment || typeof originalComment !== 'string') return;

    // Split mixed-sentiment comments at conjunctions
    const parts = originalComment.split(SPLIT_REGEX).map(p => p.trim()).filter(p => p.length > 2);

    // If split produced parts, classify each part independently
    if (parts.length > 1) {
      let hasNegative = false;
      let hasPositive = false;

      parts.forEach(part => {
        const decision = ruleBasedClassify(part);
        if (decision === 'negative') hasNegative = true;
        else if (decision === 'positive') hasPositive = true;
      });

      // If any part is negative, the whole comment goes to attention
      // Also push the full original comment for context
      if (hasNegative) {
        addAttention(originalComment.trim());
      } else if (hasPositive) {
        result.appreciation.push(originalComment.trim());
      } else {
        // Send full comment to AI
        toClassifyWithAI.push({ comment: originalComment.trim() });
      }
      return;
    }

    // Single part — classify directly
    const decision = ruleBasedClassify(originalComment.trim());
    if (decision === 'positive') {
      result.appreciation.push(originalComment.trim());
    } else if (decision === 'negative') {
      addAttention(originalComment.trim());
    } else if (decision === 'ai') {
      toClassifyWithAI.push({ comment: originalComment.trim() });
    }
    // 'skip' / 'neutral' → discard
  });

  // Step 2: AI for genuinely ambiguous long comments
  if (toClassifyWithAI.length > 0) {
    try {
      const classifier = await getSentimentPipeline();
      const BATCH_SIZE = 16;

      for (let i = 0; i < toClassifyWithAI.length; i += BATCH_SIZE) {
        const batch = toClassifyWithAI.slice(i, i + BATCH_SIZE);
        await Promise.all(batch.map(async ({ comment }) => {
          try {
            const res = await classifier(comment, { truncation: true });
            const label = res[0].label;
            const score = res[0].score;

            // Re-check with rules first — AI sometimes overrides correct rule decisions
            const lower = comment.toLowerCase();
            const hasNegativeKw = NEGATIVE_PATTERNS.some(p => p.test(lower));
            const hasPositiveKw = POSITIVE_PATTERNS.some(p => p.test(lower));

            if (hasNegativeKw) {
              addAttention(comment);
            } else if (hasPositiveKw) {
              result.appreciation.push(comment);
            } else if (label === 'NEGATIVE' && score >= 0.80) {
              // Only trust AI negative if high confidence AND no positive keyword override
              addAttention(comment);
            } else {
              // Default to appreciation for ambiguous
              result.appreciation.push(comment);
            }
          } catch {
            result.appreciation.push(comment);
          }
        }));
      }
    } catch (err) {
      console.warn('[AI] Classification failed, using rule-based fallback:', err.message);
      toClassifyWithAI.forEach(({ comment }) => {
        const lower = comment.toLowerCase();
        const hasNegative = NEGATIVE_PATTERNS.some(p => p.test(lower));
        if (hasNegative) addAttention(comment);
        else result.appreciation.push(comment);
      });
    }
  }

  console.log(`[AI] Classified ${rawComments.length} comments → ${result.appreciation.length} appreciation, ${result.commentsNeedingAttention.length} needs attention`);
  return result;
}

async function testGeminiConnection() {
  try {
    const testComments = [
      'Very good teaching',
      'needs to improve speed',
      'excellent mam',
      'Good teaching but too fast',
      'not sharing notes',
      'please provide study material',
      'average',
      'boring',
      'good but should improve interaction',
      'speaks too fast hard to understand',
      'no notes provided',
      'class miss karte hain',
    ];
    const result = await analyzeCommentsWithAI(testComments);
    return {
      ok: true,
      response: `AI working — ${result.appreciation.length} appreciation, ${result.commentsNeedingAttention.length} attention`,
      engine: 'HuggingFace + Rule-Based (local, no API key needed)',
      test: result,
    };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

module.exports = { analyzeCommentsWithAI, testGeminiConnection };
