import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import pdfParse from 'pdf-parse';
import sharp from 'sharp';
import { createCanvas, loadImage } from 'canvas';
import ffmpeg from 'fluent-ffmpeg';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// --- Konfigurasi ---
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files dari folder public
app.use(express.static(path.join(__dirname, 'public')));

// KONFIGURASI API KOLOSAL AI
const API_KEY = process.env.KOLOSAL_API_KEY;
const KOLOSAL_BASE_URL = "https://api.kolosal.ai/v1";
const MODEL_NAME = "Claude Sonnet 4.5";

const UPLOAD_FOLDER = 'uploads';
const ALLOWED_EXTENSIONS = ['txt', 'pdf', 'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'];

// Buat folder uploads jika belum ada
if (!fs.existsSync(UPLOAD_FOLDER)) {
    fs.mkdirSync(UPLOAD_FOLDER, { recursive: true });
}

// Konfigurasi Multer untuk upload file
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, UPLOAD_FOLDER);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + '-' + file.originalname);
    }
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 50 * 1024 * 1024 }, // 50MB max
    fileFilter: (req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase().substring(1);
        if (ALLOWED_EXTENSIONS.includes(ext)) {
            cb(null, true);
        } else {
            cb(new Error('Format file tidak didukung'));
        }
    }
});

// --- Inisialisasi Klien AI ---
let aiClient = null;

function initializeAiClient() {
    if (!API_KEY) {
        console.log("WARNING: KOLOSAL_API_KEY tidak ditemukan di environment variables!");
        return false;
    }
    
    try {
        aiClient = new OpenAI({
            apiKey: API_KEY,
            baseURL: KOLOSAL_BASE_URL
        });
        console.log("✓ Klien Kolosal AI berhasil diinisialisasi.");
        return true;
    } catch (error) {
        console.error(`ERROR: Gagal menginisialisasi Klien AI: ${error.message}`);
        return false;
    }
}

// Inisialisasi saat startup
initializeAiClient();

// --- Helper Functions ---

function encodeImageToBase64(imagePath) {
    try {
        const imageBuffer = fs.readFileSync(imagePath);
        return imageBuffer.toString('base64');
    } catch (error) {
        console.error(`Error encoding image: ${error.message}`);
        return null;
    }
}

async function extractTextFromPdf(filePath) {
    try {
        const dataBuffer = fs.readFileSync(filePath);
        const data = await pdfParse(dataBuffer);
        return data.text;
    } catch (error) {
        console.error(`PDF Error: ${error.message}`);
        return `[Error membaca PDF: ${error.message}]`;
    }
}

async function processVideoFrames(videoPath, maxFrames = 3) {
    return new Promise((resolve, reject) => {
        const framesBase64 = [];
        const tempDir = path.join(UPLOAD_FOLDER, 'temp_frames');
        
        // Buat folder temporary untuk frames
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir, { recursive: true });
        }

        // Ambil durasi video terlebih dahulu
        ffmpeg.ffprobe(videoPath, (err, metadata) => {
            if (err) {
                console.error(`Error probing video: ${err.message}`);
                resolve([]);
                return;
            }

            const duration = metadata.format.duration;
            if (!duration || duration === 0) {
                console.error("Error: Video tidak memiliki durasi");
                resolve([]);
                return;
            }

            // Hitung timestamp untuk setiap frame
            const timestamps = [];
            for (let i = 0; i < maxFrames; i++) {
                timestamps.push((duration * (i + 1)) / (maxFrames + 1));
            }

            let processedFrames = 0;

            // Extract frames
            timestamps.forEach((timestamp, index) => {
                const outputPath = path.join(tempDir, `frame_${index}.jpg`);
                
                ffmpeg(videoPath)
                    .screenshots({
                        timestamps: [timestamp],
                        filename: `frame_${index}.jpg`,
                        folder: tempDir,
                        size: '640x360'
                    })
                    .on('end', () => {
                        try {
                            const frameBuffer = fs.readFileSync(outputPath);
                            framesBase64.push(frameBuffer.toString('base64'));
                            
                            // Hapus file frame setelah diproses
                            fs.unlinkSync(outputPath);
                            
                            processedFrames++;
                            if (processedFrames === timestamps.length) {
                                // Hapus folder temporary
                                fs.rmdirSync(tempDir);
                                resolve(framesBase64);
                            }
                        } catch (readError) {
                            console.error(`Error reading frame: ${readError.message}`);
                        }
                    })
                    .on('error', (ffmpegError) => {
                        console.error(`Error extracting frame: ${ffmpegError.message}`);
                        processedFrames++;
                        if (processedFrames === timestamps.length) {
                            resolve(framesBase64);
                        }
                    });
            });
        });
    });
}

// --- Integrasi AI ---

async function callAiApi(userText, contextData) {
    if (!aiClient) {
        return "⚠️ Maaf, sistem AI belum terkonfigurasi dengan benar. Pastikan API Key telah diatur di environment variables.";
    }

    // System Instruction
    const systemInstruction = "Anda adalah Asisten Keuangan UMKM ahli. Analisis dokumen, gambar struk, atau video kondisi toko yang diberikan. Berikan saran praktis, hemat, dan ramah. Respon dalam Bahasa Indonesia.";

    // Prepare user content
    const userContent = [{ type: "text", text: userText }];

    // Add Context Data
    if (contextData.type === 'text') {
        userContent.push({ type: "text", text: `\n\nISI DOKUMEN:\n${contextData.content}` });
    } else if (contextData.type === 'image') {
        if (contextData.content) {
            userContent.push({
                type: "image_url",
                image_url: {
                    url: `data:image/jpeg;base64,${contextData.content}`
                }
            });
        }
    } else if (contextData.type === 'video_frames') {
        if (contextData.content && contextData.content.length > 0) {
            userContent.push({ type: "text", text: "Berikut adalah beberapa frame dari video yang diunggah user:" });
            contextData.content.forEach(frame => {
                userContent.push({
                    type: "image_url",
                    image_url: {
                        url: `data:image/jpeg;base64,${frame}`
                    }
                });
            });
        }
    }

    try {
        const response = await aiClient.chat.completions.create({
            model: MODEL_NAME,
            messages: [
                { role: "system", content: systemInstruction },
                { role: "user", content: userContent }
            ],
            max_tokens: 2048,
            temperature: 0.7
        });
        
        return response.choices[0].message.content;
    } catch (error) {
        console.error(`API Error: ${error.message}`);
        return `⚠️ Maaf, terjadi kesalahan saat menghubungi AI: ${error.message}`;
    }
}

// --- Routes ---

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/health', (req, res) => {
    res.json({
        status: "healthy",
        api_configured: aiClient !== null,
        api_key_set: API_KEY !== undefined && API_KEY !== null
    });
});

app.post('/api/chat', upload.single('file'), async (req, res) => {
    try {
        const userMessage = req.body.message || '';
        
        if (!userMessage) {
            return res.status(400).json({ error: "Pesan tidak boleh kosong" });
        }
        
        const contextData = { type: 'none', content: '' };

        if (req.file) {
            const filePath = req.file.path;
            const ext = path.extname(req.file.originalname).toLowerCase().substring(1);

            // Proses berdasarkan tipe file
            if (ext === 'pdf') {
                contextData.type = 'text';
                contextData.content = await extractTextFromPdf(filePath);
            } else if (ext === 'txt') {
                contextData.type = 'text';
                try {
                    contextData.content = fs.readFileSync(filePath, 'utf-8');
                } catch (error) {
                    contextData.content = `[Gagal membaca teks file: ${error.message}]`;
                }
            } else if (['jpg', 'jpeg', 'png'].includes(ext)) {
                contextData.type = 'image';
                const encoded = encodeImageToBase64(filePath);
                if (encoded) {
                    contextData.content = encoded;
                } else {
                    // Hapus file sebelum return error
                    fs.unlinkSync(filePath);
                    return res.status(400).json({ error: "Gagal membaca gambar" });
                }
            } else if (['mp4', 'mov', 'avi'].includes(ext)) {
                const frames = await processVideoFrames(filePath, 3);
                if (frames && frames.length > 0) {
                    contextData.type = 'video_frames';
                    contextData.content = frames;
                } else {
                    // Hapus file sebelum return error
                    fs.unlinkSync(filePath);
                    return res.status(400).json({ error: "❌ Gagal membaca video" });
                }
            }

            // Hapus file setelah diproses untuk hemat storage
            try {
                fs.unlinkSync(filePath);
            } catch (error) {
                console.warn(`Warning: Gagal menghapus file ${filePath}: ${error.message}`);
            }
        }

        const aiReply = await callAiApi(userMessage, contextData);

        res.json({ reply: aiReply });
        
    } catch (error) {
        console.error(`Error in chat endpoint: ${error.message}`);
        res.status(500).json({
            error: "Terjadi kesalahan pada server",
            message: error.message
        });
    }
});

// Error handler untuk multer
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: 'File terlalu besar. Maksimal 50MB' });
        }
        return res.status(400).json({ error: error.message });
    } else if (error) {
        return res.status(400).json({ error: error.message });
    }
    next();
});

// Start server
const PORT = process.env.PORT || 8000;
app.listen(PORT, '0.0.0.0', () => {
    console.log(`✓ Server berjalan di port ${PORT}`);
    console.log(`✓ Environment: ${process.env.NODE_ENV || 'development'}`);
});