# Test checklist

Do this list locally, again after deployment, and once more on the morning of the demo.

## 0. System

| Do | Expected |
|---|---|
| Open the Streamlit URL | Title appears, sidebar: **FastAPI backend connected**, **Gemini API key set** (cloud: "Mode: ... embedded") |
| (Two-terminal or Docker mode only) open http://localhost:8000/docs | FastAPI Swagger page with groups 2.1 to 2.5 |

## 1. Gemini auto-reply

| Do | Expected |
|---|---|
| Webhook test → keep the sample text → **Send webhook** | Green "Saved. Status: replied" and a short polite AI reply |
| Send a Korean comment, e.g. `배송은 얼마나 걸리나요?` | Reply in Korean |
| Comment dashboard → **Refresh** | Table shows the comments; counters Total / Replied update |
| Choose a comment in "Comment detail" | JSON with `status`, `ai_reply`, `received_at` |
| **Generate the reply again** | `ai_reply` changes |
| System settings → **Test Gemini connection** | Green message from Gemini |
| (Optional, local) `/docs` → `POST /webhook/comment` → Try it out | 200 response with the saved record (shows the webhook works without the UI) |

## 2. Data analysis & ML

| Do | Expected |
|---|---|
| Titanic EDA tab | Metrics (891 / 418 rows, 38.4% survival), "test.csv has 'Survived'? No", missing-value chart (Age, Cabin, Embarked), age histogram, survival by sex / class / age, correlation heatmap |
| MovieLens EDA tab | 1,000,209 ratings, rating distribution, age chart, genre chart, gender tables |
| Model comparison tab | Table of 6 models with Recall, F1, AUROC, PR-AUC; bar chart; "Selected model: XGBoost" |
| Final performance table | Hold-out metrics (about Recall 0.65, F1 0.71, AUROC 0.82, PR-AUC 0.79, accuracy 0.79) |
| **Download all test predictions (CSV)** | File with 418 rows: PassengerId, Survived |
| Form: class 1, female, age 29 → **Predict survival** | "Survived" with high probability (about 97%) |
| Form: class 3, male, age 25 | "Did not survive" with low probability |

## 3. CNN (ResNet-18)

| Do | Expected |
|---|---|
| Image URL of a golden retriever → **Classify image** | "Prediction: golden retriever (xx%)" + blue "It looks like a DOG or a CAT!" + top-5 chart |
| Upload a cat photo | A cat class (tabby, tiger cat, Egyptian cat, Persian cat, Siamese cat) + blue message |
| Image of a boat / car | Red "Classified as another class." |
| Bad URL (e.g. `https://example.com/x.jpg`) | Red error "Could not load image ..." (no crash) |

## 4. YOLO (YOLOv8n)

| Do | Expected |
|---|---|
| Upload a photo with a dog and a cat → **Detect objects** | Annotated image with boxes; table like dog 0.68, cat 0.88, chair 0.65 |
| Move the confidence slider to 0.6 and detect again | Fewer boxes |
| Street photo | person, car, traffic light, etc. |

## 5. RAG

| Do | Expected |
|---|---|
| Upload a text PDF (e.g. a course syllabus) → **Build the vector database** | "Ready: file.pdf -> N chunks" |
| Ask a question whose answer is in the file | Correct short answer + source chunks with page numbers |
| Ask something not in the file (e.g. "Who won the World Cup?") | "I could not find this in the document." |
| Change top-k slider | Number of source chunks changes |

## Demo-day safety

- Open the app 5 minutes early so it wakes up.
- Run each tab once before you present (models load, then everything is fast).
- Keep test files on your desktop: dog+cat photo, golden retriever URL, one short PDF.
- Screenshot every tab as a backup in case the Wi-Fi fails.
