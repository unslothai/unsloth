#!/bin/bash

# 汉化安装/执行脚本
# Unsloth Studio 本地化

echo "=== Unsloth Studio 汉化包 ==="
echo ""

cd studio/frontend

echo "1. 安装 i18n 依赖..."
npm install react-i18next i18next i18next-browser-languagedetector

echo ""
echo "2. 检查翻译文件..."
if [ -f "src/i18n/config.ts" ]; then
    echo "✓ i18n 配置文件存在"
else
    echo "✗ i18n 配置文件中缺失"
    exit 1
fi

if [ -f "src/i18n/locales/zh-CN.json" ] && [ -f "src/i18n/locales/en-US.json" ]; then
    echo "✓ 翻译文件存在"
else
    echo "✗ 翻译文件中缺失"
    exit 1
fi

if [ -f "src/components/ui/language-toggle.tsx" ]; then
    echo "✓ 语言切换器组件存在"
else
    echo "✗ 语言切换器组件中缺失"
    exit 1
fi

echo ""
echo "3. 启动开发服务器..."
echo "   访问 http://localhost:5173"
npm run dev
