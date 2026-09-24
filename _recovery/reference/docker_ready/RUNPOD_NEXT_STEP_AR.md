# الخطوة التالية — RunPod Serverless

ارفع الملفات التالية في root الريبو:

- Dockerfile
- requirements-serverless.txt
- .dockerignore

قبل عمل Build تأكد أن لديك بالضبط:

```text
checkpoints/
  specialist/
    checkpoint-44/
      adapter_config.json
      adapter_model.safetensors
  meta/
    checkpoint-18/
      adapter_config.json
      adapter_model.safetensors

prompts/
  advisors/
    advisor_01.md
    ...
    advisor_35.md
  meta/
    AOS-META-00.md

advisors/
  advisors_registry_35.json

handler_advisory_council.py
```

مهم:
الـ Dockerfile سيفشل عمدًا إذا كان adapter_model.safetensors مجرد Git LFS pointer
بدل الملف الحقيقي.

بعد رفع الملفات:
1. Commit إلى branch main.
2. افتح RunPod.
3. اذهب إلى Serverless / Builds أو GitHub build الخاص بالريبو الحالي.
4. اختر repo: da02bod-art/athar-qwen-training
5. branch: main
6. Dockerfile path: Dockerfile
7. ابدأ Build.

لا تنشئ Endpoint جديد قبل نجاح الـBuild.

عند نجاح الـBuild، احتفظ باسم/Tag الصورة الناتجة ثم استخدمها في الـServerless Endpoint.
