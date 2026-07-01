---
title: FarmRisk
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# FarmRisk

FarmRisk is an AI-powered agro-meteorological advisory backend built with FastAPI.

## Features

- 🌦️ Weather forecast APIs
- 🤖 AI-generated crop advisories
- 📍 Village and location resolution
- 🌱 Crop-specific recommendations
- 🔍 Semantic search using Sentence Transformers (`BAAI/bge-small-en-v1.5`)
- ⚡ Production-ready FastAPI backend

## Tech Stack

- FastAPI
- Python 3.12.1
- Sentence Transformers
- Hugging Face Transformers
- PyTorch
- Uvicorn

## Deployment

This project is configured to run as a **Docker Space** on Hugging Face.

The application starts automatically using the provided `Dockerfile`.
