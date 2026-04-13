# Unsloth Studio 汉化包说明

## 📦 已完成的汉化部分

### ✅ 核心框架
- i18n 配置文件 (`src/i18n/config.ts`)
- English/中文翻译文件 (`src/i18n/locales/`)
- 语言切换器组件 (`src/components/ui/language-toggle.tsx`)

### ✅ 界面汉化
- **导航栏** - Studio, Recipes, Export, Chat, Update 等菜单项
- **认证界面** - 登录，修改密码表单
- **Studio 训练主界面** - 标签页，标题，状态文本
- **Chat 聊天设置面板** - 模型、采样、工具、偏好设置等参数

---

## 🔧 使用方法

```bash
# 1. 安装依赖
cd studio/frontend
npm install react-i18next i18next i18next-browser-languagedetector

# 2. 启动开发服务器
npm run dev

# 3. 访问应用 (默认 http://localhost:5173)
# 点击页面右上角语言选择器切换到"简体中文"
```

---

## 🚀 当前支持的语言

| 语言 | 代码 | 状态 |
|------|------|------|
| 英语 | en-US | ✅ 完成 (基准语言) |
| 简体中文 | zh-CN | ✅ 主要界面完成 |

---

## 📝 已完成的界面清单

### 1. 导航栏 (navbar.tsx)
- [x] Studio/Recipes/Export/Chat 导航项
- [x] 更新提示弹窗
- [x] 移动端菜单
- [x] 主题切换按钮

### 2. 认证界面 (auth-form.tsx)
- [x] 登录表单
- [x] 修改密码表单  
- [x] 错误消息
- [x] 表单标签

### 3. Studio 训练界面 (studio-page.tsx)
- [x] 页面标题：微调工作室
- [x] 配置/当前运行/历史记录 标签页
- [x] 加载状态文本

### 4. Chat 聊天设置 (chat-settings-sheet.tsx)
- [x] 预设管理（保存/删除）
- [x] 系统提示词
- [x] 模型设置
- [x] 采样参数（温度/Top P/Top K/Min P 等）
- [x] 工具设置
- [x] 偏好设置

---

## ⏳ 后续可扩展汉化的模块

以下是可以进一步汉化的模块（需要时可用）:

### Training (训练界面)
- `features/training/hooks/use-training-runtime-lifecycle.ts`
- `features/studio/sections/model-section.tsx`
- `features/studio/sections/dataset-section.tsx`  
- `features/studio/sections/params-section.tsx`
- `features/studio/sections/progress-section.tsx`

### Export (导出界面)
- `features/export/export-page.tsx`

### Data Recipes (数据配方)
- `features/data-recipes/*.tsx`

---

## 💡 添加新翻译

编辑翻译文件:
- `src/i18n/locales/en-US.json` - 英文
- `src/i18n/locales/zh-CN.json` - 中文

在组件中使用:
```typescript
import { useTranslation } from 'react-i18next';

function MyComponent() {
  const { t } = useTranslation();
  
  return (
    <div>
      <h1>{t("myKey")}</h1>
      <p>{t("anotherKey", { param1: value })}</p>
    </div>
  );
}
```

---

## 🌐 常见问题

**Q: 语言切换后部分文本仍是英文？**
A: 这些区域可能未完成汉化，请检查是否需要添加到翻译词典中。

**Q: 如何贡献更多翻译？**  
A: 直接编辑 `zh-CN.json` 翻译文件即可。

**Q: 是否有热更新？**
A: 使用 `npm run dev` 时支持热更新，翻译修改后会立即生效。

---

**🎉 感谢使用 Unsloth Studio 中文版!**
