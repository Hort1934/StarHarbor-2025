# Team Photos Setup Guide

## 📸 How to Add Team Photos

### Method 1: Base64 Encoding (Current Implementation)
The current HTML uses placeholder base64 data. To add real photos:

1. **Resize photos to 160x160 pixels** for optimal display
2. **Convert to base64** using online tool or command:
   ```bash
   # Linux/Mac
   base64 -i photo.jpg
   
   # Windows PowerShell
   [convert]::ToBase64String((Get-Content -Path "photo.jpg" -Encoding byte))
   ```
3. **Replace the base64 strings** in `frontend/index.html`

### Method 2: External Image Files (Recommended)
1. **Create images folder:**
   ```
   frontend/
   ├── images/
   │   ├── team-ihor.jpg
   │   ├── team-alina.jpg
   │   └── team-veronika.jpg
   └── index.html
   ```

2. **Update HTML src attributes:**
   ```html
   <img src="images/team-ihor.jpg" class="team-avatar" alt="Ihor Marchenko">
   <img src="images/team-alina.jpg" class="team-avatar" alt="Alina Koyun">
   <img src="images/team-veronika.jpg" class="team-avatar" alt="Veronika Horobets">
   ```

### Method 3: GitHub Assets (For GitHub Pages)
1. **Upload photos to GitHub repository**
2. **Get raw image URLs**
3. **Update HTML with GitHub URLs:**
   ```html
   <img src="https://raw.githubusercontent.com/Hort1934/StarHarbor-2025/main/docs/team/ihor.jpg" class="team-avatar" alt="Ihor Marchenko">
   ```

## 📋 Current Team Information

- **Ihor Marchenko** (@hort19345) - Team Lead & Backend Developer
- **Alina Koyun** (@alinakoyun) - Frontend Developer & UI/UX
- **Veronika Horobets** (@horobets) - Project Manager & Documentation

All team members are from Ukraine 🇺🇦

## 🎨 Photo Requirements

- **Size**: 160x160 pixels (square format)
- **Format**: JPG or PNG
- **Quality**: High resolution, professional appearance
- **Background**: Preferably clean or blurred
- **File size**: Under 100KB each for web optimization