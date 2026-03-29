require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs-extra');
const path = require('path');
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');

const app = express();
const PORT = process.env.PORT || 3000;

// --- ミドルウェア ---
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// --- ファイルアップロード設定（拡張子を保持） ---
const storage = multer.diskStorage({
  destination: 'uploads/',
  filename: (req, file, cb) => {
    const ext = file.originalname.split('.').pop();
    cb(null, Date.now() + '.' + ext);
  }
});
const upload = multer({
  storage: storage,
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB
  fileFilter: (req, file, cb) => {
    const allowed = ['audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/m4a',
                     'audio/x-m4a', 'audio/webm', 'video/mp4', 'video/webm',
                     'audio/ogg', 'audio/flac', 'audio/x-flac'];
    if (allowed.includes(file.mimetype) || file.originalname.match(/\.(mp3|wav|m4a|mp4|webm|ogg|flac|mpeg|mpga)$/i)) {
      cb(null, true);
    } else {
      cb(new Error('対応していないファイル形式です。MP3/WAV/M4A/MP4/WebM/OGG/FLACに対応しています。'));
    }
  }
});

// --- APIクライアント初期化 ---
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// --- 簡易利用回数管理（本番ではDB使用） ---
const usageMap = new Map(); // IP → { count, resetDate }

const DAILY_FREE_LIMIT = parseInt(process.env.DAILY_FREE_LIMIT || '3');

function checkUsage(ip) {
  const today = new Date().toDateString();
  const usage = usageMap.get(ip);
  if (!usage || usage.resetDate !== today) {
    usageMap.set(ip, { count: 0, resetDate: today });
    return { allowed: true, remaining: DAILY_FREE_LIMIT };
  }
  if (usage.count >= DAILY_FREE_LIMIT) {
    return { allowed: false, remaining: 0 };
  }
  return { allowed: true, remaining: DAILY_FREE_LIMIT - usage.count };
}

function incrementUsage(ip) {
  const today = new Date().toDateString();
  const usage = usageMap.get(ip) || { count: 0, resetDate: today };
  usage.count++;
  usage.resetDate = today;
  usageMap.set(ip, usage);
}

// =============================================
// API: 文字起こし（Whisper）
// =============================================
app.post('/api/transcribe', upload.single('audio'), async (req, res) => {
  const ip = req.headers['x-forwarded-for'] || req.ip;

  try {
    // 利用制限チェック
    const usage = checkUsage(ip);
    if (!usage.allowed) {
      return res.status(429).json({
        error: '本日の無料利用回数を超えました。',
        remaining: 0
      });
    }

    if (!req.file) {
      return res.status(400).json({ error: '音声ファイルが見つかりません。' });
    }

    console.log(`[文字起こし開始] ${req.file.originalname} (${(req.file.size / 1024 / 1024).toFixed(1)}MB)`);

    // Whisper APIで文字起こし
    const transcription = await openai.audio.transcriptions.create({
      file: fs.createReadStream(req.file.path),
      model: 'whisper-1',
      language: 'ja',
      response_format: 'verbose_json',
      timestamp_granularities: ['segment']
    });

    // アップロードファイル削除
    await fs.remove(req.file.path);

    console.log(`[文字起こし完了] ${transcription.text.length}文字`);

    res.json({
      text: transcription.text,
      segments: transcription.segments || [],
      duration: transcription.duration || 0
    });

  } catch (err) {
    console.error('[文字起こしエラー]', err.message);
    if (req.file) await fs.remove(req.file.path).catch(() => {});
    res.status(500).json({ error: '文字起こしに失敗しました: ' + err.message });
  }
});

// =============================================
// API: 要約（Claude）
// =============================================
app.post('/api/summarize', async (req, res) => {
  const ip = req.headers['x-forwarded-for'] || req.ip;

  try {
    const { transcript } = req.body;
    if (!transcript || transcript.trim().length === 0) {
      return res.status(400).json({ error: '文字起こしテキストが空です。' });
    }

    console.log(`[要約開始] ${transcript.length}文字`);

    const message = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 2000,
      messages: [
        {
          role: 'user',
          content: `あなたは優秀な議事録作成アシスタントです。以下の会議の文字起こしテキストを分析し、必ず以下のJSON形式で要約してください。JSON以外のテキストは一切含めないでください。

出力フォーマット:
{
  "title": "会議のタイトル（内容から推測）",
  "date": "今日の日付",
  "participants": ["発言者1", "発言者2"],
  "summary": "会議全体の概要（2〜3文）",
  "keyPoints": ["要点1", "要点2", "要点3"],
  "decisions": ["決定事項1", "決定事項2"],
  "risks": ["リスクや懸念事項1"],
  "actions": [
    {"who": "担当者", "what": "タスク内容", "deadline": "期限"}
  ]
}

注意:
- 発言者名が不明な場合は「参加者A」「参加者B」としてください
- 決定事項がない場合は空配列にしてください
- リスクがない場合も空配列にしてください
- アクションアイテムがない場合も空配列にしてください

文字起こしテキスト:
${transcript}`
        }
      ]
    });

    // Claude応答をパース
    let summaryText = message.content[0].text;
    // ```json ``` で囲まれている場合に対応
    summaryText = summaryText.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();

    let summary;
    try {
      summary = JSON.parse(summaryText);
    } catch (parseErr) {
      console.error('[JSONパースエラー]', summaryText);
      return res.status(500).json({ error: '要約の生成に失敗しました。もう一度お試しください。' });
    }

    // 利用回数カウント（文字起こし＋要約で1回）
    incrementUsage(ip);
    const usage = checkUsage(ip);

    console.log(`[要約完了]`);

    res.json({
      summary,
      remaining: usage.remaining
    });

  } catch (err) {
    console.error('[要約エラー]', err.message);
    res.status(500).json({ error: '要約に失敗しました: ' + err.message });
  }
});

// --- 利用状況確認 ---
app.get('/api/usage', (req, res) => {
  const ip = req.headers['x-forwarded-for'] || req.ip;
  const usage = checkUsage(ip);
  res.json({ remaining: usage.remaining, limit: DAILY_FREE_LIMIT });
});

// --- uploadsフォルダ作成 ---
fs.ensureDirSync('uploads');

// --- サーバー起動 ---
app.listen(PORT, () => {
  console.log(`✅ 議事録AI サーバー起動: http://localhost:${PORT}`);
  console.log(`   Whisper API: ${process.env.OPENAI_API_KEY ? '✓ 設定済み' : '✗ 未設定'}`);
  console.log(`   Claude API:  ${process.env.ANTHROPIC_API_KEY ? '✓ 設定済み' : '✗ 未設定'}`);
  console.log(`   1日の無料回数: ${DAILY_FREE_LIMIT}回`);
});
