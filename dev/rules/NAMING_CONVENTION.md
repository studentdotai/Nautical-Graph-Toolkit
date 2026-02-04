# Naming Convention Guide - Vector Nautical & Nautical-Graph-Toolkit

## 📋 Official Names

### **Vector Nautical**
- **Type:** Company / Organization name
- **Legal Entity:** OpenCollective organization
- **Use When:** Discussing company strategy, mission, funding model, advocacy efforts

**Correct Usage Examples:**
- ✅ "Vector Nautical operates on a self-funded model"
- ✅ "Vector Nautical's mission includes advocating for open S-57 data"
- ✅ "Vector Nautical explores strategic partnerships"
- ❌ "Vector Nautical calculates routes" (wrong - the product does this, not the company)

---

### **Nautical-Graph-Toolkit (NGT)**
- **Type:** Product / Library name
- **Current Version:** v0.1.1 (January 2026)
- **Status:** Open-source Python library (AGPL-3.0)
- **Use When:** Discussing technical features, architecture, code, performance

**Correct Usage Examples:**
- ✅ "The Nautical-Graph-Toolkit transforms S-57 data into routing graphs"
- ✅ "Nautical-Graph-Toolkit leverages PostGIS for spatial queries"
- ✅ "NGT v0.1.1 supports three graph types: Base, Fine, and H3"
- ✅ "the toolkit" (acceptable shorthand when context is clear)
- ❌ "NGT advocates for open data" (wrong - that's Vector Nautical's mission)

**Acceptable Variations:**
- "Nautical Graph Toolkit" (without hyphen - acceptable in prose)
- "NGT" (abbreviation - use after first mention)
- "the toolkit" (shorthand - use when context is clear)

---

### **Route Assistant**
- **Type:** Product Vision / Future Product Name
- **Version:** NGT 1.0 (not yet released)
- **Description:** Integrated solution = Backend (NGT) + UI + ML capabilities
- **Use When:** Discussing long-term vision, complete product roadmap, thesis concept

**Correct Usage Examples:**
- ✅ "Route Assistant (NGT 1.0) represents the complete vision from the Master's thesis"
- ✅ "The long-term goal is to evolve NGT into Route Assistant"
- ✅ "Route Assistant will combine routing backend with advanced UI and ML"
- ❌ "Route Assistant is currently available" (wrong - it's a vision, not current)
- ❌ "Route Assistant v0.1.1" (wrong - current product is NGT v0.1.1)

**Important:** Route Assistant refers to the **future state**, not the current product.

---

## 🎯 Usage Decision Tree

```
Question: Are you talking about...

├─ Company/Organization/Strategy?
│  └─ Use: "Vector Nautical"
│
├─ Current Product/Library/Code?
│  └─ Use: "Nautical-Graph-Toolkit" or "NGT" or "the toolkit"
│
├─ Future Vision (Backend + UI + ML)?
│  └─ Use: "Route Assistant (NGT 1.0)"
│
└─ OpenCollective Funding?
   └─ Use: "Vector Nautical's OpenCollective"
```

---

## ✅ Correct Examples by Context

### **Company Strategy & Mission**
```
Vector Nautical operates on a sustainable self-funded model, developing the
Nautical-Graph-Toolkit part-time while advocating for open maritime data access.
```

### **Technical Product Description**
```
The Nautical-Graph-Toolkit (NGT) is an open-source Python library that converts
S-57 ENC data into routing graphs. The toolkit leverages PostGIS for spatial
indexing and NetworkX for graph algorithms.
```

### **Roadmap & Vision**
```
The current Nautical-Graph-Toolkit (v0.1.1) provides core routing functionality.
Vector Nautical's roadmap includes QGIS integration (2026) and evolving toward
Route Assistant (NGT 1.0), a comprehensive decision support system combining
the routing backend with advanced UI and ML capabilities.
```

### **Funding Discussion**
```
Vector Nautical explores three funding pathways: (1) Community support via
OpenCollective for accelerating Nautical-Graph-Toolkit development,
(2) Strategic partnerships for enterprise features, (3) Dual-licensing models.
```

---

## ❌ Common Mistakes to Avoid

| ❌ Wrong | ✅ Correct | Why |
|---------|----------|-----|
| "Vector Nautical calculates routes" | "Nautical-Graph-Toolkit calculates routes" | Company vs. Product |
| "NGT advocates for open data" | "Vector Nautical advocates for open data" | Product doesn't have mission, company does |
| "Route Assistant v0.1.1" | "Nautical-Graph-Toolkit v0.1.1" | Route Assistant is future vision, not current |
| "the toolkit's mission" | "Vector Nautical's mission" | Toolkit doesn't have mission, company does |
| "Vector Nautical's graph algorithms" | "Nautical-Graph-Toolkit's graph algorithms" | Algorithms belong to product, not company |

---

## 📝 Document Consistency Checklist

When writing about the project, verify:

- [ ] "Vector Nautical" used only for company/organization/mission context
- [ ] "Nautical-Graph-Toolkit" (or "NGT" or "the toolkit") used for product/technical context
- [ ] "Route Assistant" used only for future vision (NGT 1.0), marked as roadmap
- [ ] OpenCollective always associated with "Vector Nautical" (not product)
- [ ] Version numbers always with product name (NGT v0.1.1, not "Vector Nautical v0.1.1")

---

## 🎯 Quick Reference

**Company:** Vector Nautical
**Current Product:** Nautical-Graph-Toolkit (NGT) v0.1.1
**Future Vision:** Route Assistant (NGT 1.0)
**Funding Platform:** Vector Nautical's OpenCollective

**Rule of Thumb:**
- If it's about people, strategy, or mission → **Vector Nautical**
- If it's about code, features, or technology → **Nautical-Graph-Toolkit**
- If it's about the complete future product → **Route Assistant**

---

## 📚 Related Documentation

- [Project Documentation Standards](./DOCUMENTATION.md)
- [Code Standards](./CODE_STANDARDS.md)
- [Claude AI Agent Guidelines](./AGENTS.md)