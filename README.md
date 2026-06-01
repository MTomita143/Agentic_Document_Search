# Agentic Document Search

🔎 **Find Documents the Way You Remember Them**

*"The PDF with the blue graph."*  
*"The investor deck from last quarter."*  
*"That report mentioning Apache."*

Search documents from vague memories, not filenames.

---

## ✨ What Makes It Different?

Traditional search:

```text
Query
→ Retrieve
→ Return Results
```

Agentic Document Search:

```text
Memory Clue
→ Narrow Candidates
→ Read Relevant Files
→ Check Visual Evidence
→ Explain Why It Matched
```

The agent investigates documents the way a person would.


## 📂 Supported Files

- 📄 PDF
- 📊 PowerPoint
- 📝 Word
- 📈 Excel

---

## 🤖 Search Modes

### 🎯Auto Search
Agent chooses the best strategy.

### ⚡Filename Search
Search filenames and folder structure only.

### 📦Content Search
Read candidate files and search their contents.

### 👁️Visual Search
Find documents using visual clues such as graphs, charts, layouts, screenshots, and diagrams.

---

## ☁️ Azure-Powered Architecture

```text
User
 ↓
Streamlit UI
 ↓
Semantic Kernel Agent
 ↓
Azure OpenAI
 ↓
Metadata Search
 ↓
Content Analysis
 ↓
Visual Verification
 ↓
Ranked Results
```

### Azure Services Used

- Azure OpenAI
- Azure AI Vision
- Azure AI Translator
- Azure Blob Storage
- Azure App Service
- Semantic Kernel

---

## 🚀 Why This Matters

Employees rarely remember filenames.

They remember:

- 📊 a graph
- 📁 a folder
- 🏢 a company name
- 📄 part of a report
- 🎨 a visual layout

Agentic Document Search helps users find documents the way they actually remember them.

---

## 🏃 Run Locally

```bash
streamlit run src/ui.py
```