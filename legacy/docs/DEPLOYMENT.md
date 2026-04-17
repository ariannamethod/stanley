# HuggingFace Space Deployment Guide

Quick guide for deploying STANLEY to HuggingFace Spaces.

---

## Prerequisites

- HuggingFace account
- Git installed locally
- Repository cloned

---

## Files Needed for Space

The following files are ready for deployment:

```
Required Files:
├── app.py                    # Main Gradio application
├── requirements_space.txt    # Rename to requirements.txt for Space
├── README_SPACE.md          # Rename to README.md for Space
├── origin.txt               # Stanley's identity seed (34KB)
├── stanley/                 # Weightless architecture (copy entire folder)
└── stanley_hybrid/          # Hybrid mode (copy entire folder)
```

---

## Deployment Steps

### 1. Create New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in details:
   - **Name**: `stanley-demo` (or your choice)
   - **License**: `gpl-3.0`
   - **SDK**: `Gradio`
   - **Hardware**: `CPU basic` (sufficient for weightless mode)
     - For hybrid mode with GPT-2: `CPU upgrade` or `T4 small` recommended
   - **Visibility**: Public (or Private for testing)

### 2. Configure Space Metadata

Create or edit `README.md` in your Space with this header:

```yaml
---
title: STANLEY — Weightless Architecture Demo
emoji: 🧠
colorFrom: purple
colorTo: black
sdk: gradio
sdk_version: 4.19.0
app_file: app.py
pinned: false
license: gpl-3.0
---
```

Then add the content from `README_SPACE.md`.

### 3. Prepare Files

```bash
# In your stanley repository
cd /path/to/stanley

# Copy files to a staging directory
mkdir -p ../stanley-space
cp app.py ../stanley-space/
cp origin.txt ../stanley-space/
cp requirements_space.txt ../stanley-space/requirements.txt
cp README_SPACE.md ../stanley-space/README.md
cp -r stanley ../stanley-space/
cp -r stanley_hybrid ../stanley-space/
cp LICENSE ../stanley-space/
```

### 4. Push to HuggingFace

```bash
cd ../stanley-space

# Initialize git (if not already)
git init

# Add HuggingFace remote (replace USERNAME and SPACE_NAME)
git remote add origin https://huggingface.co/spaces/USERNAME/SPACE_NAME

# Add files
git add .

# Commit
git commit -m "Initial deployment of STANLEY v1.0"

# Push
git push origin main
```

Alternatively, use the HuggingFace web interface:
1. Upload files directly via browser
2. Or use `huggingface-cli` tool

### 5. Monitor Build

1. Go to your Space URL
2. Watch the "Building..." status
3. First build may take 5-10 minutes (downloading dependencies)
4. Hybrid mode will download distilgpt2 (~250MB) on first run

---

## Configuration Options

### Hardware Selection

**CPU Basic (Free):**
- ✅ Weightless mode works perfectly
- ⚠️ Hybrid mode may be slow (10-15s per response)
- RAM: 16GB (sufficient)

**CPU Upgrade ($):**
- ✅ Both modes work well
- ✅ Hybrid mode faster (~5s per response)
- RAM: 32GB

**GPU T4 Small ($$):**
- ✅ Optimal for hybrid mode (~2-3s per response)
- ✅ Can handle multiple concurrent users
- VRAM: 16GB

**Recommendation:** Start with CPU Basic, upgrade if needed.

### Environment Variables

No environment variables needed. All configuration is in code.

Optional variables (set in Space settings):
```
GRADIO_SERVER_NAME=0.0.0.0  # Default
GRADIO_SERVER_PORT=7860     # Default
```

---

## Testing Your Deployment

### 1. Check Logs

In Space interface:
- Click "Logs" tab
- Watch for:
  - ✅ "STANLEY initialized"
  - ✅ "Weightless mode: ✅"
  - ✅ "Hybrid mode: ✅" (if torch/transformers loaded)

### 2. Test Weightless Mode

1. Select "Weightless" mode
2. Enter prompt: "Tell me about yourself"
3. Click "Generate Response"
4. Should see response in <2s
5. Check metrics display updates

### 3. Test Hybrid Mode (if available)

1. Select "Hybrid" mode
2. Enter prompt: "What is memory?"
3. Click "Generate Response"
4. First run may be slow (downloading GPT-2)
5. Subsequent runs should be faster

### 4. Test Example Prompts

Click example buttons to verify they populate input.

---

## Troubleshooting

### Build Fails

**Symptom:** Build stuck or fails

**Causes:**
1. Missing dependencies in `requirements.txt`
2. Syntax error in `app.py`
3. Out of memory during build

**Solutions:**
- Check logs for specific error
- Verify `requirements.txt` is correct
- Test locally: `gradio app.py`

### App Crashes on Launch

**Symptom:** Space shows error page

**Common issues:**
1. `origin.txt` missing
2. Import errors (stanley/ not found)
3. Out of memory

**Solutions:**
- Verify all folders copied correctly
- Check file structure in Space files tab
- Upgrade hardware tier if OOM

### Hybrid Mode Disabled

**Symptom:** "Hybrid mode not available"

**Causes:**
1. torch/transformers failed to install
2. Out of memory loading GPT-2
3. Network error downloading model

**Solutions:**
- Check build logs for torch installation
- Upgrade to CPU Upgrade or GPU tier
- Models cache in `/home/user/.cache/huggingface/`

### Slow Response Times

**Symptom:** >10s per response

**Solutions:**
- Upgrade hardware tier
- First run is always slower (model loading)
- Use distilgpt2 (already configured)
- Disable hybrid mode for faster responses

---

## Optimization Tips

### Reduce Space Size

If deploying without hybrid mode:

1. Remove hybrid dependencies from `requirements.txt`:
   ```diff
   - torch>=2.0.0
   - transformers>=4.30.0
   ```

2. Remove `stanley_hybrid/` folder

3. Modify `app.py`:
   ```python
   HYBRID_AVAILABLE = False  # Force disable
   ```

Result: ~100MB Space vs ~500MB

### Speed Up First Load

Models are cached after first download. Subsequent loads are fast.

To pre-cache (advanced):
1. Use persistent storage tier
2. Or accept 30-60s first-load latency

### Handle Multiple Users

For production with traffic:
1. Use GPU tier
2. Consider implementing queue system
3. Add rate limiting if needed

---

## Post-Deployment

### Update README

Add Space badge to main repository README:

```markdown
### 🚀 Try It Live

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/USERNAME/SPACE_NAME)
```

### Monitor Usage

HuggingFace provides:
- Usage analytics
- Error logs
- User feedback (via Community tab)

### Iterate

Space updates when you push to git:
```bash
# Make changes locally
git add .
git commit -m "Update: description"
git push origin main

# Space rebuilds automatically
```

---

## Alternative: Docker Deployment

For non-HF deployment:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_space.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t stanley-demo .
docker run -p 7860:7860 stanley-demo
```

---

## Support

Issues with deployment?

1. **Check HF Docs:** https://huggingface.co/docs/hub/spaces
2. **Check logs** in Space interface
3. **Test locally** with `gradio app.py`
4. **File issue** on GitHub repository

---

## Checklist

Before deployment:

- [ ] `app.py` tested locally
- [ ] `requirements_space.txt` complete
- [ ] `README_SPACE.md` ready
- [ ] `origin.txt` included
- [ ] `stanley/` folder complete
- [ ] `stanley_hybrid/` included (if hybrid mode)
- [ ] License file added
- [ ] Space created on HF
- [ ] Metadata configured
- [ ] Files pushed to Space
- [ ] Build successful
- [ ] Both modes tested
- [ ] Example prompts work
- [ ] Metrics display correctly

---

**Ready to deploy!** 🚀

**STANLEY v1.0 — Architecture > Parameters**

🧠 💫 🔺
