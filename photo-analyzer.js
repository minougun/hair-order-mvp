const EPS = 1e-7;
const BINS = 16;

const VALUE_LABELS = {
  length: { short: "短め", medium: "中くらい", long: "長め" },
  volume: { low: "少なめ", normal: "普通", high: "多め" },
  stiffness: { soft: "柔らかい", normal: "普通", hard: "硬い" },
  wave: { straight: "直毛", wavy: "うねり", curly: "強いクセ" },
  crown: { weak: "弱い", normal: "普通", strong: "強い" },
};

export async function analyzeHairAttributes(files) {
  const front = await analyzeFile(files.front);
  const side = await analyzeFile(files.side);
  const top = await analyzeFile(files.top);

  if (!front && !side && !top) {
    throw new Error("写真が選択されていません");
  }

  const length = estimateLength(front, side);
  const volume = estimateVolume(top, front, side);
  const stiffness = estimateStiffness(front, side, top);
  const wave = estimateWave(front, side, top);
  const crown = estimateCrown(top);

  const predictions = {
    length: length.value,
    volume: volume.value,
    stiffness: stiffness.value,
    wave: wave.value,
    crown: crown.value,
  };

  const confidence = {
    length: length.confidence,
    volume: volume.confidence,
    stiffness: stiffness.confidence,
    wave: wave.confidence,
    crown: crown.confidence,
  };

  const notes = [
    formatNote("length", length),
    formatNote("volume", volume),
    formatNote("stiffness", stiffness),
    formatNote("wave", wave),
    formatNote("crown", crown),
  ];

  return {
    predictions,
    confidence,
    notes,
    usedPhotos: {
      front: Boolean(front),
      side: Boolean(side),
      top: Boolean(top),
    },
  };
}

function formatNote(key, result) {
  const label = VALUE_LABELS[key]?.[result.value] ?? result.value;
  return `${label}（信頼度${Math.round(result.confidence * 100)}% / ${result.reason}）`;
}

async function analyzeFile(file) {
  if (!file) return null;
  const imageData = await readImageData(file);
  return extractFeatures(imageData);
}

async function readImageData(file) {
  const bitmap = await createImageBitmap(file);
  const maxSide = 512;
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(48, Math.round(bitmap.width * scale));
  const height = Math.max(48, Math.round(bitmap.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close?.();
  return ctx.getImageData(0, 0, width, height);
}

function extractFeatures(imageData) {
  const { width, height, data } = imageData;
  const size = width * height;

  const gray = new Float32Array(size);
  let graySum = 0;
  let graySqSum = 0;

  for (let i = 0; i < size; i += 1) {
    const idx = i * 4;
    const r = data[idx] / 255;
    const g = data[idx + 1] / 255;
    const b = data[idx + 2] / 255;
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    gray[i] = luma;
    graySum += luma;
    graySqSum += luma * luma;
  }

  const mean = graySum / size;
  const variance = Math.max(0, graySqSum / size - mean * mean);
  const contrast = clamp01(Math.sqrt(variance) * 2.2);
  const darkness = clamp01(1 - mean);

  const strongThreshold = 0.18;
  const edgeThreshold = 0.1;
  const orientationBins = new Array(BINS).fill(0);
  const rowStrongCounts = new Uint32Array(height);

  const midLeft = Math.floor(width * 0.2);
  const midRight = Math.ceil(width * 0.8);

  const minSide = Math.min(width, height);
  const cx = width / 2;
  const cy = height / 2;
  const crownInnerR = minSide * 0.08;
  const crownOuterR = minSide * 0.3;

  let edgePixels = 0;
  let strongPixels = 0;
  let centerTotal = 0;
  let centerStrong = 0;
  let centerAlignMag = 0;
  let centerMagSum = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const p = y * width + x;
      const gx =
        -gray[p - width - 1] + gray[p - width + 1]
        - 2 * gray[p - 1] + 2 * gray[p + 1]
        - gray[p + width - 1] + gray[p + width + 1];
      const gy =
        -gray[p - width - 1] - 2 * gray[p - width] - gray[p - width + 1]
        + gray[p + width - 1] + 2 * gray[p + width] + gray[p + width + 1];

      const mag = Math.sqrt(gx * gx + gy * gy) / 4;
      if (mag > edgeThreshold) {
        edgePixels += 1;

        const ori = normalizeOrientation(Math.atan2(gy, gx));
        const bin = Math.min(BINS - 1, Math.floor((ori / Math.PI) * BINS));
        orientationBins[bin] += mag;
      }

      if (mag > strongThreshold) {
        strongPixels += 1;
        if (x >= midLeft && x <= midRight) {
          rowStrongCounts[y] += 1;
        }
      }

      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist >= crownInnerR && dist <= crownOuterR) {
        centerTotal += 1;
        if (mag > edgeThreshold) {
          centerStrong += 1;
          const ori = normalizeOrientation(Math.atan2(gy, gx));
          const tangent = normalizeOrientation(Math.atan2(dy, dx) + Math.PI / 2);
          const diff = orientationDiff(ori, tangent);
          const align = 1 - diff / (Math.PI / 2);
          centerAlignMag += align * mag;
          centerMagSum += mag;
        }
      }
    }
  }

  const area = Math.max(1, (width - 2) * (height - 2));
  const edgeDensity = clamp01(edgePixels / area);
  const strongEdgeDensity = clamp01(strongPixels / area);
  const orientationEntropy = normalizedEntropy(orientationBins);
  const orientationCoherence = orientationMaxShare(orientationBins);

  const rowSignalThreshold = Math.max(3, Math.floor((midRight - midLeft + 1) * 0.03));
  let lastActiveRow = Math.floor(height * 0.52);
  for (let y = 0; y < height; y += 1) {
    if (rowStrongCounts[y] > rowSignalThreshold) {
      lastActiveRow = y;
    }
  }
  const bottomActiveRatio = clamp01(lastActiveRow / Math.max(1, height - 1));

  const centerEdgeDensity = centerTotal > 0 ? clamp01(centerStrong / centerTotal) : 0;
  const crownSpiralScore = centerMagSum > EPS ? clamp01(centerAlignMag / centerMagSum) : 0;

  return {
    width,
    height,
    darkness,
    contrast,
    edgeDensity,
    strongEdgeDensity,
    orientationEntropy,
    orientationCoherence,
    bottomActiveRatio,
    centerEdgeDensity,
    crownSpiralScore,
  };
}

function estimateLength(front, side) {
  const usable = [front, side].filter(Boolean);
  if (usable.length === 0) {
    return { value: "short", confidence: 0.2, reason: "正面・横写真が未入力" };
  }

  const bottom = average(usable.map((f) => f.bottomActiveRatio));
  let value = "short";
  if (bottom > 0.76) value = "long";
  else if (bottom > 0.62) value = "medium";

  const signal = clamp01((bottom - 0.45) / 0.38);
  const confidence = clamp(usable.length === 2 ? 0.6 + signal * 0.28 : 0.42 + signal * 0.28, 0.3, 0.92);

  return {
    value,
    confidence,
    reason: `毛先の到達位置を画像高さの${Math.round(bottom * 100)}%付近と推定`,
  };
}

function estimateVolume(top, front, side) {
  const source = top || front || side;
  if (!source) {
    return { value: "normal", confidence: 0.2, reason: "画像が未入力" };
  }

  const score = clamp01(source.strongEdgeDensity * 0.52 + source.darkness * 0.32 + source.contrast * 0.16);

  let value = "normal";
  if (score > 0.33) value = "high";
  else if (score < 0.21) value = "low";

  const base = top ? 0.66 : 0.44;
  const confidence = clamp(base + Math.abs(score - 0.27) * 1.2, 0.3, 0.9);

  return {
    value,
    confidence,
    reason: `髪領域の密度スコア=${score.toFixed(2)}`,
  };
}

function estimateStiffness(front, side, top) {
  const source = blendedFeature([front, side, top]);
  if (!source) {
    return { value: "normal", confidence: 0.2, reason: "画像が未入力" };
  }

  const score = clamp01(source.orientationCoherence * 0.5 + source.contrast * 0.25 + (1 - source.orientationEntropy) * 0.25);

  let value = "normal";
  if (score > 0.62) value = "hard";
  else if (score < 0.44) value = "soft";

  const confidence = clamp(0.46 + Math.abs(score - 0.53) * 1.05, 0.3, 0.88);

  return {
    value,
    confidence,
    reason: `毛流れの直線性スコア=${score.toFixed(2)}`,
  };
}

function estimateWave(front, side, top) {
  const source = blendedFeature([front, side, top]);
  if (!source) {
    return { value: "straight", confidence: 0.2, reason: "画像が未入力" };
  }

  const score = clamp01(source.orientationEntropy * 0.65 + (1 - source.orientationCoherence) * 0.35);

  let value = "straight";
  if (score > 0.72) value = "curly";
  else if (score > 0.56) value = "wavy";

  const confidence = clamp(0.48 + Math.abs(score - 0.5) * 0.95, 0.3, 0.9);

  return {
    value,
    confidence,
    reason: `うねり量スコア=${score.toFixed(2)}`,
  };
}

function estimateCrown(top) {
  if (!top) {
    return { value: "normal", confidence: 0.22, reason: "頭頂写真が未入力" };
  }

  const score = clamp01(top.crownSpiralScore * 0.68 + top.centerEdgeDensity * 0.32);

  let value = "normal";
  if (score > 0.44) value = "strong";
  else if (score < 0.24) value = "weak";

  const confidence = clamp(0.56 + Math.abs(score - 0.34) * 1.0, 0.35, 0.9);

  return {
    value,
    confidence,
    reason: `つむじ渦スコア=${score.toFixed(2)}`,
  };
}

function blendedFeature(features) {
  const valid = features.filter(Boolean);
  if (valid.length === 0) return null;

  const totalWeight = valid.length;
  return {
    orientationCoherence: average(valid.map((f) => f.orientationCoherence), totalWeight),
    orientationEntropy: average(valid.map((f) => f.orientationEntropy), totalWeight),
    contrast: average(valid.map((f) => f.contrast), totalWeight),
  };
}

function normalizeOrientation(angle) {
  let value = angle % Math.PI;
  if (value < 0) value += Math.PI;
  return value;
}

function orientationDiff(a, b) {
  const d = Math.abs(a - b);
  return Math.min(d, Math.PI - d);
}

function normalizedEntropy(values) {
  const total = values.reduce((sum, v) => sum + v, 0);
  if (total <= EPS) return 0;

  let entropy = 0;
  for (const value of values) {
    if (value <= EPS) continue;
    const p = value / total;
    entropy -= p * Math.log2(p);
  }
  return clamp01(entropy / Math.log2(values.length));
}

function orientationMaxShare(values) {
  const total = values.reduce((sum, v) => sum + v, 0);
  if (total <= EPS) return 0;
  const max = Math.max(...values);
  return clamp01(max / total);
}

function average(values, divisorOverride) {
  if (values.length === 0) return 0;
  const divisor = divisorOverride ?? values.length;
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, divisor);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clamp01(value) {
  return clamp(value, 0, 1);
}
