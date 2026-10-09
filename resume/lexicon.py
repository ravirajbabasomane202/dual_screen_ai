"""Technology lexicon used to classify skills found in a resume."""
from __future__ import annotations

import re

def _m(s: str) -> dict[str, str]:
    out = {}
    for item in [x.strip() for x in s.split(",") if x.strip()]:
        out[item.lower()] = item
    return out

PROGRAMMING_LANGUAGES = _m("""Python, Java, JavaScript, TypeScript, C++, C#, C, Go, Golang, Rust, Kotlin, Swift, Ruby, PHP, Scala, R,
MATLAB, Perl, Dart, Objective-C, Lua, Haskell, Elixir, Clojure, Groovy, Bash, Shell, PowerShell, SQL, PL/SQL, T-SQL, HTML, CSS, SASS,
Assembly, VHDL, Verilog, Julia, Fortran, COBOL, Solidity, Visual Basic, VBA, F#""")
FRAMEWORKS = _m("""Django, Flask, FastAPI, Spring, Spring Boot, Hibernate, React, React Native, Angular, Vue, Vue.js, Svelte, Next.js, Nuxt,
Node.js, Express, NestJS, ASP.NET, .NET, Laravel, Rails, Ruby on Rails, Symfony, jQuery, Bootstrap, Tailwind, Redux, Flutter,
TensorFlow, PyTorch, Keras, scikit-learn, Scikit-Learn, Pandas, NumPy, SciPy, Matplotlib, OpenCV, Hugging Face, Transformers, LangChain,
LlamaIndex, spaCy, NLTK, XGBoost, LightGBM, YOLO, Spark, PySpark, Hadoop, Kafka, Airflow, Celery, Pydantic, SQLAlchemy, gRPC, GraphQL,
Qt, PySide, PyQt, Tkinter, Electron, JUnit, Pytest, Selenium, Jest, Cypress, Streamlit, Gradio, Dash, Plotly, Django REST Framework""")
TOOLS = _m("""Git, GitHub, GitLab, Bitbucket, Docker, Kubernetes, Terraform, Ansible, Jenkins, CircleCI, GitHub Actions, CI/CD, Linux, Unix,
Windows, macOS, AWS, Azure, GCP, Google Cloud, Heroku, Vercel, EC2, S3, Lambda, PostgreSQL, MySQL, SQLite, MongoDB, Redis, Elasticsearch,
Cassandra, DynamoDB, Oracle, SQL Server, Snowflake, BigQuery, Redshift, Tableau, Power BI, Excel, Jira, Confluence, Postman, Swagger,
Nginx, Apache, RabbitMQ, Grafana, Prometheus, Datadog, Splunk, Jupyter, VS Code, IntelliJ, PyCharm, Figma, Ollama, MLflow, DVC, Kibana,
Webpack, Maven, Gradle, npm, Yarn, Pip, Conda, Vagrant, Helm, Istio, Sentry, Selenium, Wireshark, ROS, Unity, Unreal Engine, Arduino,
Raspberry Pi, Android Studio, Xcode, Salesforce, SAP""")
AMBIGUOUS = {"go", "r", "c", "swift", "spring", "express", "dash", "excel", "unity", "oracle", "windows", "transformers", "shell"}
_CONTEXT = re.compile(r"(?:[,;|/•·●▪]\s*$|^\s*[,;|/•·●▪]|\blanguages?\b|\bprogramming\b|\bproficient\b|\bskills?\b)", re.I)

_ALL = [("programming_languages", PROGRAMMING_LANGUAGES), ("frameworks", FRAMEWORKS), ("tools", TOOLS)]


def _pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9+#])" + re.escape(term) + r"(?![A-Za-z0-9+#]|\.[A-Za-z])", re.I)


_PATTERNS = {(cat, key): _pattern(key) for cat, table in _ALL for key in table}


def find_terms(text: str) -> dict[str, list[str]]:
    """Return {'programming_languages': [...], 'frameworks': [...], 'tools': [...]} (display names, de-duplicated)."""
    out: dict[str, list[str]] = {"programming_languages": [], "frameworks": [], "tools": []}
    for (cat, key), pat in _PATTERNS.items():
        for m in pat.finditer(text):
            if key in AMBIGUOUS:
                if not text[m.start()].isupper():
                    continue
                before = text[max(0, m.start() - 3):m.start()]
                after = text[m.end():m.end() + 3]
                line = text[text.rfind("\n", 0, m.start()) + 1: (text.find("\n", m.end()) if text.find("\n", m.end()) != -1 else len(text))]
                listy = line.count(",") + line.count("|") + line.count("•") >= 2
                if not (listy or re.search(r"[,;|/•]", before + after) or re.search(r"\b(language|programming|skills?)\b", line, re.I)):
                    continue
            disp = dict(_ALL)[cat][key]
            if disp not in out[cat]:
                out[cat].append(disp)
            break
    # a term may sit in several tables (Selenium): keep the first category only
    seen: set[str] = set()
    for cat in out:
        out[cat] = [t for t in out[cat] if not (t.lower() in seen or seen.add(t.lower()))]
    return out
