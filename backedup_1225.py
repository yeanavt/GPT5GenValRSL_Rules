#!/usr/bin/env python3
"""
JetBrains Inspection Rule Generator and Validator

This script processes a CSV file containing JetBrains inspection data and:
1. Extracts @annotations from inspection data and saves to JSON
2. Generates RSL rules using GPT-4 (leveraging builtins.json, RSL syntax, and existing rule examples)
3. Creates rule descriptions using GPT-4
4. Finds relevant 3rd-party web pages using GPT-5 with web_search tool (effort: high)
5. Validates URLs with content relevance scoring (annotation-weighted + LLM verification)
6. Evaluates the generated rules using GPT-5.2 as a judge

Models Used (per user request):
GPT-5

Supported Frameworks (27):
 # 'jpa-and-kotlin': [
    #     ('https://kotlinlang.org/docs/jpa.html', 'Kotlin JPA Documentation'),
    #     ('https://spring.io/guides/tutorials/spring-boot-kotlin/', 'Spring Boot with Kotlin Guide'),
    #     ('https://www.baeldung.com/kotlin/jpa', 'Baeldung Kotlin JPA Tutorial'),
    #],
# # # # JPA-and-Kotlin ----> Not included # # # # 
CDI, JPA, Spring Boot, Spring Security, Spring Core, Spring Cloud Stream,
Spring Data, Spring Integration, Spring MVC, Spring Modulith, Spring AOP, JAX-RS, Hibernate,
AOP, Javadoc, Java EE, Bean Validation, Micronaut, Micronaut Data, Quarkus, JavaFX,
JUnit, Lombok, Serialization Issues, TestNG

References:
- GPT-5.2 Announcement: https://openai.com/index/introducing-gpt-5-2/
- Web Search Tool: https://platform.openai.com/docs/guides/tools-web-search
"""

import os
import csv
import json
import time
import re
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Error: openai package not installed. Run: pip install openai>=1.0.0")

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    print("Warning: requests/beautifulsoup4 not installed. URL validation disabled.")
    print("Run: pip install requests beautifulsoup4")

# =============================================================================
# MODEL CONFIGURATION (Per User Requirements)
# =============================================================================

# GPT-4 for rule generation  
# Evaluations from developers and benchmarks show (12/21/2025):
# GPT-4's training and evaluation included programming and reasoning benchmarks emphasizing code correctness.
# Empirical developer feedback
# Developers who have compared:
# Report that GPT-4 produces more reliable unit tests
# Report that GPT-4 is better at inferring method contracts and exception behaviors
# Report that GPT-4o can occasionally hallucinate or generate less precise test assertions

MODEL_RULE_GENERATION = "gpt-5"

# GPT-5 for web search with effort: high (User Request: GPT-5)
MODEL_WEB_SEARCH = "gpt-5"

# GPT-5 for evaluation/judging (User Request: GPT-5)
MODEL_EVALUATION = "gpt-5"

# Temperature settings
# Temperature = 0 for GPT-4 to minimize randomness in rule generation
TEMPERATURE_GPT4 = 0

# Web Search Configuration
# WEB_SEARCH_REASONING_EFFORT = "high"
# WEB_SEARCH_CONTEXT_SIZE = "high"

# URL Validation Configuration
URL_VALIDATION_MAX_URLS = 2          # Maximum validated URLs to return
URL_VALIDATION_MIN_RELEVANCE = 0.30  # Minimum relevance score (0.0-1.0)
URL_VALIDATION_TIMEOUT = 15          # HTTP request timeout in seconds
URL_VALIDATION_USE_LLM = True        # Use GPT-4 for borderline cases

# =============================================================================
# RSL (Rule Specification Language) CORE SYNTAX
# =============================================================================

RSL_SYNTAX = r"""
Specification := Rule Id Body
Body := '{' Stmt Stmt* '}'
Stmt := ForStmt | IfStmt | AssertStmt | DeclStmt ';'

ForStmt := 'for' '(' Type Id 'in' Exp ')' Body
IfStmt := 'if' '(' Exp ')' Body

AssertStmt := 'assert' '(' Exp ')' '{' MsgStmt ';' '}'
MsgStmt := 'msg' '(' ',' SimExp (',' SimExp)* ')'

DeclStmt := Type Id '=' Exp

Exp := SimExp
     | SimExp AND Exp
     | SimExp OR  Exp
     | NOT Exp

SimExp := Id
        | Lit
        | FunctionCall
        | '(' Exp ')'
        | FunctionCall '==' SimExp
        | exists '(' Type Id in Exp ')' '(' Exp ')'

Type := '⟨' Id '⟩' | file | class | method | field | String
Lit := StringLit | CharLit | IntLit | FloatLit
FunctionCall := Id '(' Params ')'
Params := SimExp (',' SimExp)*
"""

# =============================================================================
# MANUALLY CHECKED RELEVANT WEBPAGES PER FRAMEWORK
# =============================================================================

FALLBACK_WEBPAGES = {
    
    'cdi': [
        ('https://jakarta.ee/specifications/cdi/4.1/jakarta-cdi-spec-4.1', 'Jakarta CDI Specification'),
        ('https://docs.jboss.org/weld/reference/latest/en-US/html/part4.html', 'Weld CDI Reference'),
        ('https://www.baeldung.com/java-ee-cdi', 'Baeldung Java EE CDI Tutorial'),
    ],
    'jpa': [
        ('https://jakarta.ee/specifications/persistence/3.2/jakarta-persistence-spec-3.2', 'Jakarta Persistence Specification'),
        ('https://docs.oracle.com/javaee/7/tutorial/persistence-intro.htm', 'Oracle JPA Tutorial'),
        ('https://www.baeldung.com/learn-jpa-hibernate', 'Baeldung JPA/Hibernate Guide'),
    ],
    'spring boot': [
        ('https://docs.spring.io/spring-boot/documentation.html', 'Spring Boot Documentation'),
        ('https://spring.io/guides', 'Official Spring Guides'),
        ('https://www.baeldung.com/spring-boot', 'Baeldung Spring Boot Tutorials'),
    ],
    'spring security': [
        ('https://www.geeksforgeeks.org/advance-java/spring-security-annotations/', 'Spring Security Annotations'),
        ('https://spring.io/guides/topicals/spring-security-architecture', 'Spring Security Architecture'),
        ('https://www.baeldung.com/spring-security-method-security', 'Baeldung Spring Security Tutorials'),
    ],
    'spring core': [
        ('https://www.javacodegeeks.com/2019/05/spring-core-annotations.html', 'Spring Core Annotations'),
        ('https://unsekhablecom.wordpress.com/2018/11/02/15-spring-core-annotation-examples/', 'Spring Core Examples'),
        ('https://www.baeldung.com/spring-core-annotations', 'Baeldung Spring Core Annotations'),
    ],

    'spring cloud stream': [
        ('https://docs.spring.io/spring-cloud-stream/docs/Brooklyn.SR1/reference/htmlsingle/', 'Spring Cloud Stream Reference'),
        ('https://developer.okta.com/blog/2020/04/15/spring-cloud-stream', 'Spring Cloud Stream'),
        ('https://www.baeldung.com/spring-cloud-stream', 'Baeldung Spring Cloud Stream'),
    ],
    'spring data': [
        ('https://www.baeldung.com/spring-data-annotations', 'Spring Data Annotations'),
        ('https://spring.io/projects/spring-data', 'Spring Data Project'),
        ('https://www.baeldung.com/the-persistence-layer-with-spring-data-jpa', 'Baeldung Spring Data JPA'),
    ],
    'spring integration': [
        ('https://docs.spring.io/spring-integration/docs/current/reference/html/', 'Spring Integration Reference'),
        ('https://www.spring-doc.cn/spring-integration/6.0.9/._overview.en.html', 'Spring Integration Overview'),
        ('https://www.baeldung.com/spring-integration', 'Baeldung Spring Integration'),
    ],
    'spring mvc': [
        ('https://docs.spring.io/spring-framework/reference/web/webmvc.html', 'Spring MVC Reference'),
        ('https://www.geeksforgeeks.org/advance-java/spring-mvc-annotations-with-examples/', 'Spring MVC Annotations'),
        ('https://www.baeldung.com/spring-mvc-tutorial', 'Baeldung Spring MVC Tutorial'),
    ],
    'spring modulith': [
        ('https://docs.spring.io/spring-modulith/docs/current/api/org/springframework/modulith/events/ApplicationModuleListener.html', 'Spring Modulith Reference'),
        ('https://spring.io/projects/spring-modulith', 'Spring Modulith Project'),
        ('https://www.baeldung.com/spring-modulith', 'Baeldung Spring Modulith'),
    ],
    'spring aop': [
        ('https://docs.spring.io/spring-framework/reference/core/aop/ataspectj.html', 'Spring AOP AspectJ'),
        ('https://www.geeksforgeeks.org/java/spring-aop-with-examples/', 'Spring Guides'),
        ('https://mkyong.com/spring3/spring-aop-aspectj-annotation-example/', 'Baeldung Spring AOP Tutorial'),
    ],
    'jax-rs': [
        ('https://jakarta.ee/specifications/restful-ws/4.0/jakarta-restful-ws-spec-4.0', 'Jakarta RESTful Web Services Specification'),
        ('https://docs.oracle.com/javaee/7/tutorial/jaxrs003.htm#GIPZZ', 'Oracle JAX-RS Tutorial'),
        ('https://www.baeldung.com/rest-with-spring-series', 'Baeldung REST with Spring'),
    ],

    'hibernate': [
        ('https://hibernate.org/orm/documentation/', 'Hibernate ORM Documentation'),
        ('https://docs.jboss.org/hibernate/orm/current/userguide/html_single/Hibernate_User_Guide.html', 'Hibernate User Guide'),
        ('https://www.baeldung.com/learn-jpa-hibernate', 'Baeldung Hibernate Tutorial'),
    ],
    'aop': [
        ('https://eclipse.dev/aspectj/doc/released/progguide/index.html', 'AspectJ Programming Guide 1998-2001'),
        ('https://docs.spring.io/spring-framework/reference/core/aop/schema.html', 'Spring AOP Schema'),
        ('https://www.baeldung.com/aspectj', 'Baeldung AspectJ Tutorial'),
    ],

    'javadoc': [
        ('https://docs.oracle.com/javase/8/docs/technotes/tools/windows/javadoc.html', 'Oracle Javadoc Tool'),
        ('https://www.oracle.com/technical-resources/articles/java/javadoc-tool.html', 'How to Write Doc Comments'),
        ('https://www.baeldung.com/javadoc', 'Baeldung Javadoc Guide'),
    ],
    'java ee': [
        ('https://jakarta.ee/specifications/servlet/5.0/apidocs/', 'Jakarta EE Specifications'),
        ('https://docs.oracle.com/javaee/7/tutorial/', 'Oracle Java EE 7 Tutorial'),
        ('https://jakarta.ee/specifications/annotations/3.0/', 'Baeldung Java EE Tutorials'),
    ],
    'validation': [
        ('https://beanvalidation.org/2.0/spec/', 'Bean Validation 2.0 Specification'),
        ('https://docs.spring.io/spring-framework/reference/core/validation/beanvalidation.html', 'Spring Bean Validation'),
        ('https://www.baeldung.com/javax-validation', 'Baeldung Bean Validation Tutorial'),
    ],
    'micronaut': [
        ('https://docs.micronaut.io/latest/guide/', 'Micronaut Core Documentation'),
        ('https://guides.micronaut.io/', 'Micronaut Guides'),
        ('https://www.baeldung.com/micronaut', 'Baeldung Micronaut Tutorial'),
    ],
    'micronaut data': [
        ('https://micronaut-projects.github.io/micronaut-data/latest/guide/', 'Micronaut Data Documentation'),
        ('https://docs.micronaut.io/4.9.5/api/io/micronaut/http/annotation/PathVariable.html', 'Micronaut Data Guides'),
        ('https://dev.to/dixitgurv/microservices-design-patterns-in-java-3pfk', 'Microservices Design Patterns'),
    ],
    'quarkus': [
        ('https://quarkus.io/guides/config-reference', 'Quarkus Guides'),
        ('https://quarkus.io/guides/rest-client', 'Quarkus Documentation'),
        ('https://www.baeldung.com/quarkus-io', 'Baeldung Quarkus Tutorial'),
    ],
    'javafx': [
        ('https://docs.oracle.com/javafx/2/get_started/jfxpub-get_started.htm', 'Oracle JavaFX Get Started'),
        ('https://www.jenkov.com/tutorials/javafx/index.html', 'Jenkov JavaFX Tutorials'),
        ('https://www.jenkov.com/tutorials/javafx/fxml.html', 'JavaFX FXML')
    ],
    'junit': [
        ('https://junit.org/junit5/docs/current/user-guide/', 'JUnit 5 User Guide'),
        ('https://www.baeldung.com/junit-5', 'Baeldung JUnit 5 Guide'),
        ('https://docs.junit.org/6.0.0-RC1/user-guide/index.html', 'JUnit 6 User Guide'),
    ],
    'java-verbose': [
        ('https://www.baeldung.com/java-clean-code', 'Baeldung Clean Code'),
        ('https://refactoring.guru/refactoring/catalog', 'Refactoring Catalog'),
        ('https://dzone.com/articles/introduction-to-lombok', 'Introduction to Lombok'),
    ],
    'lombok': [
        ('https://projectlombok.org/features/', 'Project Lombok Features'),
        ('https://www.baeldung.com/intro-to-project-lombok', 'Baeldung Lombok Introduction'),
        ('https://projectlombok.org/features/', 'Project Lombok Features'),
    ],

    'serialization': [
        ('https://stackoverflow.com/questions/63783474/what-is-the-use-of-serial-annotation-as-of-java-14', 'What is the use of serial annotation as of Java 14?'),
        ('https://www.baeldung.com/java-14-serial-annotation', 'Baeldung Java 14 Serial Annotation'),
        ('https://dzone.com/articles/javas-serial-annotation', 'Java Serial Annotation'),
    ],

    'testng': [
        ('https://testng.org/', 'TestNG Documentation'),
        ('https://www.baeldung.com/testng', 'Baeldung TestNG Tutorial'),
        ('https://www.tutorialspoint.com/testng/testng_quick_guide.htm', 'TestNG Quick Guide'),
    ],
}

# =============================================================================
# FILE PATHS CONFIGURATION
# =============================================================================



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV_PATH = os.path.join(BASE_DIR, "c_input.csv")  # Default to c_input.csv if found
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "sample_output_with_rules.csv")
SAMPLE_INPUT_CSV_PATH = os.path.join(BASE_DIR, "sample_input.csv")
BUILTINS_JSON_PATH = os.path.join(BASE_DIR, "builtinfs.json")
ANNOTATIONS_JSON_PATH = os.path.join(BASE_DIR, "extracted_annotations.json")
URL_REPORT_JSON_PATH = os.path.join(BASE_DIR, "url_report.json")
EXISTING_RULES_PATH_WINDOWS = os.path.join(os.path.dirname(BASE_DIR), "artifact-submission", "rules")

DELAY_BETWEEN_ROWS = 2
MAX_ROWS_TO_PROCESS = None


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RowData:
    """Data class to hold parsed row information"""
    framework: str
    source: str
    topic: str
    description: str
    examples: str


@dataclass
class JetBrainsData:
    """Data class to hold extracted annotations for a single rule"""
    row_index: int
    framework: str
    topic: str
    description: str
    annotations: Dict[str, List[str]] = field(default_factory=dict)
    annotation_count: int = 0
    keywords: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.annotations:
            self.annotations = {
                'all': [],
                'from_topic': [],
                'from_issue_description': [],
                'from_examples': []
            }
        if not self.keywords:
            self.keywords = {
                'from_the_name_of_framework': [],
                'from_topic': [],
                'from_issue_description': []
            }


@dataclass
class GeneratedContent:
    """Data class to hold generated content for each row"""
    rule: str
    rule_description: str
    web_pages: str
    none_existing_functions: str
    evaluation: str


# =============================================================================
# RULE MAKER JB CLASS
# =============================================================================

class RuleMakerJB:
    """
    Extracts and manages @annotations from JetBrains inspection data.
    Saves to JSON for use in relevance scoring.
    """
    
    def __init__(self, json_path: str = None):
        self.json_path = json_path or ANNOTATIONS_JSON_PATH
        self.annotations_data: Dict[int, JetBrainsData] = {}
        self._load_existing_json()
    
    def _load_existing_json(self):
        """Load existing annotations JSON if it exists"""
        try:
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for rule in data.get('rules', []):
                        idx = rule['row_index']
                        self.annotations_data[idx] = JetBrainsData(
                            row_index=idx,
                            framework=rule.get('framework', ''),
                            topic=rule.get('topic', ''),
                            description=rule.get('description', ''),
                            annotations=rule.get('annotations', {}),
                            annotation_count=rule.get('annotation_count', 0)
                        )
                print(f"Loaded {len(self.annotations_data)} existing annotation records from JSON")
        except Exception as e:
            print(f"Note: Starting fresh annotations JSON: {e}")
    
    def extract_annotations(self, text: str) -> List[str]:
        """
        Extract all @annotations from text.
        Handles: @Entity, @Column(name="id"), @javax.persistence.Entity
        """
        if not text:
            return []
        
        # Pattern matches @Word and @package.path.Word
        full_pattern = r'(@(?:[\w.]+\.)?[\w]+)'
        full_matches = re.findall(full_pattern, text)
        
        # Deduplicate while preserving order
        all_annotations = []
        seen = set()
        
        for anno in full_matches:
            if anno not in seen:
                seen.add(anno)
                all_annotations.append(anno)
            
            # Also add short form (e.g., @javax.persistence.Entity → @Entity)
            short_form = '@' + anno.split('.')[-1].lstrip('@')
            if short_form not in seen and short_form != anno:
                seen.add(short_form)
                all_annotations.append(short_form)
        
        return all_annotations
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        if not text:
            return []
            
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                  'by', 'from', 'as', 'into', 'through', 'during', 'before',
                  'after', 'above', 'below', 'between', 'under', 'that', 'which',
                  'who', 'whom', 'this', 'these', 'those', 'reports', 'errors', 
                  'error', 'such', 'inspection', 'check', 'checks', 'report',
                  'jetbrains', 'issue', 'problem', 'detect', 'detects'}
        
        # Split by non-word chars
        words = [w for w in re.split(r'\W+', text.lower()) 
                   if w and w not in stop_words and len(w) > 2 and not w.isdigit()]
        
        # Deduplicate while preserving order
        return list(dict.fromkeys(words))
    
    def extract_and_store(self, row_index: int, row_data: RowData) -> JetBrainsData:
        """Extract annotations and keywords from all fields and store"""
        # Annotations
        topic_annotations = self.extract_annotations(row_data.topic)
        desc_annotations = self.extract_annotations(row_data.description)
        example_annotations = self.extract_annotations(row_data.examples) if row_data.examples else []
        
        # Keywords
        framework_keywords = self.extract_keywords(row_data.framework)
        topic_keywords = self.extract_keywords(row_data.topic)
        desc_keywords = self.extract_keywords(row_data.description)
        
        # Combine all unique annotations
        all_annotations = []
        seen = set()
        for anno in topic_annotations + desc_annotations + example_annotations:
            if anno not in seen:
                seen.add(anno)
                all_annotations.append(anno)
        
        anno_data = JetBrainsData(
            row_index=row_index,
            framework=row_data.framework,
            topic=row_data.topic,
            description=row_data.description,
            annotations={
                'all': all_annotations,
                'from_topic': topic_annotations,
                'from_issue_description': desc_annotations,
                'from_examples': example_annotations
            },
            annotation_count=len(all_annotations),
            keywords={
                'from_the_name_of_framework': framework_keywords,
                'from_topic': topic_keywords,
                'from_issue_description': desc_keywords
            }
        )
        
        self.annotations_data[row_index] = anno_data
        return anno_data
    
    def save_to_json(self):
        """Save all extracted annotations and keywords to JSON file with detailed schema"""
        all_unique_items = set()
        
        for ad in self.annotations_data.values():
            # Add all annotations
            all_unique_items.update(ad.annotations.get('all', []))
            # Add all keywords from all sources
            if hasattr(ad, 'keywords'):
                for key_list in ad.keywords.values():
                    all_unique_items.update(key_list)
        
        output = {
            'total_number_keywords_including_unique_annotations': len(all_unique_items),
            'all_unique_annotations': sorted(list(all_unique_items)),
            'rules': []
        }
        
        for idx in sorted(self.annotations_data.keys()):
            ad = self.annotations_data[idx]
            rule_entry = {
                'row_index_from_the_input_file': ad.row_index,
                'annotations': ad.annotations,
                'keywords_other_than_@annotation_from_topic': ad.keywords
            }
            output['rules'].append(rule_entry)
        
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.annotations_data)} rule records to: {self.json_path}")
        return self.json_path
    
    def get_annotations_for_row(self, row_index: int) -> List[str]:
        """Get all annotations for a specific row"""
        if row_index in self.annotations_data:
            return self.annotations_data[row_index].annotations.get('all', [])
        return []


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class JetBrainsRuleGenerator:
    """Main class for generating and validating rules from JetBrains inspection data"""
    
    EXCLUDED_DOMAINS = [
        'jetbrains.com', 'jetbrains.cn', 'jetbrains.net',
        'intellij.com', 'youtrack.jetbrains.com'
    ]
    
    VALID_DOC_DOMAINS = [
        'docs.oracle.com', 'docs.jboss.org', 'hibernate.org',
        'spring.io', 'docs.spring.io', 'baeldung.com',
        'stackoverflow.com', 'jakarta.ee', 'eclipse.org', 'eclipse.dev',
        'apache.org', 'github.com', 'mkyong.com', 'tutorialspoint.com',
        'geeksforgeeks.org', 'javatpoint.com', 'dzone.com',
        'vogella.com', 'journaldev.com', 'howtodoinjava.com',
        'micronaut.io', 'docs.micronaut.io', 'guides.micronaut.io',
        'quarkus.io', 'openjfx.io', 'junit.org', 'testng.org',
        'projectlombok.org', 'beanvalidation.org',
        'refactoring.guru', 'jenkov.com', 'micronaut-projects.github.io'
    ]
    
    def __init__(self, api_key: Optional[str] = None, 
                 builtins_path: str = BUILTINS_JSON_PATH,
                 rules_path: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required.")
        self.client = OpenAI(api_key=self.api_key)
        self.builtins = self._load_builtins(builtins_path)
        self.existing_rules = self._load_existing_rules(rules_path)
        self.ruleMakerJB = RuleMakerJB(ANNOTATIONS_JSON_PATH)
        
        print(f"Loaded {len(self.builtins)} RSL built-in functions")
        print(f"Loaded {len(self.existing_rules)} existing rule examples")
    
    def _load_builtins(self, builtins_path: str) -> List[Dict]:
        
        try:
            if os.path.exists(builtins_path):
                with open(builtins_path, 'r', encoding='utf-8') as f:
                    return json.load(f).get('builtinfs', [])
        except Exception as e:
            print(f"Warning: Error loading builtinfs.json: {e}")
        return []
    
    def _load_existing_rules(self, rules_path: str = None) -> List[Dict[str, str]]:
        existing_rules = []
        if rules_path is None:
            if os.path.exists(EXISTING_RULES_PATH_WINDOWS):
                rules_path = EXISTING_RULES_PATH_WINDOWS
      
            else:
                return []
        
        try:
            if os.path.exists(rules_path):
                for txt_file in glob.glob(os.path.join(rules_path, "*.txt")):
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            existing_rules.append({
                                'filename': os.path.basename(txt_file),
                                'content': f.read()
                            })
                    except:
                        continue
        except:
            pass
        return existing_rules
    
    def _format_builtins_for_prompt(self) -> str:
        if not self.builtins:
            return "No RSL builtins available."
        
        formatted = "=== RSL Built-in Functions ===\n\n"
        categories = {}
        for b in self.builtins:
            cat = b.get('category', 'other')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(b)
        
        for category, funcs in categories.items():
            formatted += f"## {category.upper()}:\n"
            for func in funcs:
                formatted += f"  - {func['name']}: {func['purpose']}\n    Signature: {func.get('signature', 'N/A')}\n    Return: {func.get('return', 'N/A')}\n"
            formatted += "\n"
        return formatted
    
    def _format_existing_rules_for_prompt(self) -> str:
        if not self.existing_rules:
            return "No existing rule examples available."
        
        formatted = "=== Existing RSL Rule Examples ===\n\n"
        for i, rule in enumerate(self.existing_rules, 1):
            formatted += f"--- Example {i}: {rule['filename']} ---\n{rule['content']}\n\n"
        return formatted
    
    def _format_builtins_list(self) -> str:
        """Format a concise list of valid RSL built-in function names for evaluation."""
        if not self.builtins:
            return "No RSL built-in functions available."
        
        function_names = [b.get('name', '') for b in self.builtins if b.get('name')]
        return "Valid RSL Built-in Functions:\n" + ", ".join(function_names)
    
    # =========================================================================
    # RULE GENERATION
    # =========================================================================
    
    def generate_rule(self, row_data: RowData) -> str:
        """Generate an RSL-expressed rule using GPT-5"""
        
        builtins_context = self._format_builtins_for_prompt()
        rules_context = self._format_existing_rules_for_prompt()
        
        prompt = f"""You are an expert in Java metadata bug detection using RSL (Rule Specification Language).
        Generate an RSL rule to detect the metadata bug described below.

        === RSL CORE SYNTAX ===
        {RSL_SYNTAX}

        {builtins_context}

        {rules_context}"""

        instructions = f"""

        === JETBRAINS INSPECTION DATA ===
        Framework: {row_data.framework}fb
        Source: {row_data.source}
        Topic: {row_data.topic}
        Issue Description: {row_data.description}
        Examples: {row_data.examples if row_data.examples else "No examples provided"}

        === TASK ===
        Generate a syntactically correct RSL rule that:
        1. Follows the given RSL syntax grammar exactly
        2. Uses appropriate RSL built-in functions ONLY while not introducing new built-in functions
        3. Follows patterns from the existing rule examples semantically 
        4. Refer to the provided issue description and example code snippets if any semantically and syntactically
        5. Adheres to the existing built-in functions' signature, parameters, and return value 
        6. Utimately a newly generated rule should support the usage constraints described by the topic, issue description, and example. 
        7. If the generated RSL rule is too large, split it into multiple independent RSL rules so that each rule examines only one metadata misuse pattern. 

        Output ONLY the RSL rule code."""

        messages = [
                {"role": "developer", "content": prompt}
            ]

        try:
            # response = self.client.chat.completions.create(
            #     model=MODEL_RULE_GENERATION,
            #     messages=[
            #         {"role": "system", "content": "You are a metadata bug detection expert using RSL (Rule Specification Language). Generate syntactically and semantically correct RSL-expressed rules."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=TEMPERATURE_GPT4,
            #     #max_tokens=1500  - - - -> GPT-4 model used, but not GPT-5 models 
            # )
            # return response.choices[0].message.content.strip()
            print(" --->>> GENERATING RULES with GPT-5!! <<<---")
            resp = self.client.responses.create(
                model=MODEL_RULE_GENERATION,
                input=messages,
                instructions=instructions,
                reasoning={"effort": "high"},
                text={"verbosity": "medium"}
            )
            result = resp.output_text.strip() if hasattr(resp, 'output_text') else "No - Evaluation failed"
            return f"{result}"


        except Exception as e:
            return f"Error generating rule: {str(e)}"

        
           
            # try:
            #     # Simpler call without tool forcing for fallback
            #     print("  STARTING with GPT-5 Eval!!...")
            #     resp = self.client.responses.create(
            #         model=MODEL_EVALUATION,
            #         input=messages,
            #         instructions=instructions,
            #         reasoning={"effort": "high"},
            #         text={"verbosity": "low"}
            #     )

            #     result = resp.output_text.strip() if hasattr(resp, 'output_text') else "No - Evaluation failed"
                
            #     # Sanitize for CSV/Excel - prevent #NAME! errors
            #     if result and result[0] in ['=', '-', '+', '@']:
            #         result = "'" + result  # Prefix with single quote to force text interpretation
                
            #     return result
            # except Exception as e:
            #     return self._fallback_evaluation(row_data, generated_rule, rule_description)
        


    
    def generate_rule_description(self, row_data: RowData, generated_rule: str) -> str:
       
        """Generate rule description using GPT-5"""
        prompt = f"""Explain this RSL-expressed rule for detecting metadata bugs Java applications:
        Based on the framework, source, topic, issue description, and examples if any
        === JETBRAINS INSPECTION DATA ===
        Framework: {row_data.framework}
        Source: {row_data.source}
        Topic: {row_data.topic}
        Issue Description: {row_data.description}
        Examples: {row_data.examples if row_data.examples else "No examples provided"}
        
        When explaining each generated rule, 
        RSL-expressed rule:
        {generated_rule}

        
        Explain this RSL-expressed rule for detecting metadata bugs Java applications."""


        instructions = f"""
      
        Provide the following information:
        1. What constraints this rule detects or each rule detects if there are multiple generated rules
        2. How and what the the existing builtin functions are used per generated rule
        3. Detection logic step by step per rule
        4. Answer if each generated rule necessarily address a bug that must be fixed for what kind of reasons"""
       
        messages = [
                {"role": "system", "content": prompt}
            ]


        try:
            # response = self.client.chat.completions.create(
            #     model=MODEL_RULE_GENERATION,
            #     messages=[
            #         {"role": "system", "content": "You are a metadata bug detection expert using RSL (Rule Specification Language). Provide explanations for the RSL rule."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=TEMPERATURE_GPT4,
            #     #max_tokens=800
            # )
            # return response.choices[0].message.content.strip()
            resp = self.client.responses.create(
                model=MODEL_RULE_GENERATION,
                input=messages,
                instructions=instructions,
                reasoning={"effort": "high"},
                text={"verbosity": "medium"}
            )
            result = resp.output_text.strip() if hasattr(resp, 'output_text') else "No - Evaluation failed"
            return f"{result}"


        except Exception as e:
            return f"Error generating description: {str(e)}"



    # ==============================================================================
    # SEARCH CANDIDATE BUILDING WITH ANNOTATION EXTRACTION and KEYWORD EXTRACTION
    # ==============================================================================
    
    def _build_search_candidates(self, row_data: RowData, row_index: int = 0) -> Tuple[List[str], JetBrainsData]:
        """
        Build search query candidates and extract/store all @annotations and keywords.
        
        Returns:
            Tuple of (search_candidates, annotation_data)
        """
        # === EXTRACT AND STORE ALL ANNOTATIONS & KEYWORDS ===
        annotation_data = self.ruleMakerJB.extract_and_store(row_index, row_data)
        all_annotations = annotation_data.annotations.get('all', [])
        
        # === GATHER KEYWORDS ===
        # Gather all keywords from the stored data
        all_keywords = []
        if hasattr(annotation_data, 'keywords'):
            # Priority: framework -> topic -> description
            for k in annotation_data.keywords.get('from_the_name_of_framework', []):
                 if k not in all_keywords: all_keywords.append(k)
            for k in annotation_data.keywords.get('from_topic', []):
                 if k not in all_keywords: all_keywords.append(k)
            for k in annotation_data.keywords.get('from_issue_description', []):
                 if k not in all_keywords: all_keywords.append(k)
        
        # Re-deduplicate
        unique_keywords = []
        seen = set()
        for k in all_keywords:
            if k not in seen:
                seen.add(k)
                unique_keywords.append(k)
        
        # Prepare keyword strings
        top_2_keywords = unique_keywords[:2]
        top_4_keywords = unique_keywords[:4]
        
        top_2_joined = ' '.join(top_2_keywords)
        top_4_joined = ' '.join(top_4_keywords)
        
        candidates = []
        
        # === GENERATE PATTERN CANDIDATES ===
        
        # 1. {annotation} AND {top_2_keywords_joined} (for each annotation)
        for anno in all_annotations:
            if top_2_joined:
                candidates.append(f"{anno} AND {top_2_joined}")
            else:
                candidates.append(f"{anno}")
        
        # 2. {top_4_keywords_joined}
        if top_4_joined:
            candidates.append(top_4_joined)
        
        # 3. {annotation} {top_4_keywords_joined} (combinations)
        for anno in all_annotations:
            if top_4_joined:
                candidates.append(f"{anno} {top_4_joined}")
        
        # Fallback if no annotations and no keywords (rare)
        if not candidates:
            base = f'{row_data.framework} {row_data.topic}'
            candidates.append(base.strip())
            
        return candidates, annotation_data

    
    # =========================================================================
    # RELEVANCE SCORING WITH ANNOTATION WEIGHTING
    # =========================================================================
    
    def _calculate_relevance_score(
        self, 
        page_text: str, 
        page_title: str, 
        row_data: RowData,
        annotation_data: JetBrainsData
    ) -> Tuple[float, dict]:
        """
        Calculate relevance score with heavy weighting on @annotations from JSON.
        
        Scoring Weights:
        - Annotation match: 4.0 per annotation (highest priority)
        - Framework name match: 3.0
        - Topic keywords: 1.5 each
        - Title bonuses: +20% if framework in title, +15% per annotation in title
        """
        page_text_lower = page_text.lower()
        page_title_lower = page_title.lower() if page_title else ""
        
        scoring_details = {
            'annotation_matches': {},
            'framework_match': 0,
            'topic_matches': {},
            'title_bonuses': []
        }
        
        total_weight = 0
        earned_weight = 0
        
        # === ANNOTATION MATCHING (Highest Priority) ===
        annotations = annotation_data.annotations.get('all', [])
        annotation_weight = 4.0
        
        for anno in annotations:
            total_weight += annotation_weight
            anno_lower = anno.lower()
            anno_without_at = anno.lstrip('@').lower()
            
            found_in = []
            if anno_lower in page_text_lower:
                found_in.append('text_exact')
            if anno_without_at in page_text_lower:
                found_in.append('text_without_at')
            if anno_lower in page_title_lower or anno_without_at in page_title_lower:
                found_in.append('title')
            
            if found_in:
                earned_weight += annotation_weight
                scoring_details['annotation_matches'][anno] = {
                    'found': True, 'locations': found_in, 'weight': annotation_weight
                }
            else:
                scoring_details['annotation_matches'][anno] = {
                    'found': False, 'locations': [], 'weight': 0
                }
        
        # === FRAMEWORK MATCHING ===
        framework = row_data.framework.lower().strip()
        framework_weight = 3.0
        total_weight += framework_weight
        
        if framework in page_text_lower:
            earned_weight += framework_weight
            scoring_details['framework_match'] = 1.0
        elif any(fw_part in page_text_lower for fw_part in framework.split() if len(fw_part) > 2):
            earned_weight += framework_weight * 0.6
            scoring_details['framework_match'] = 0.6
        
        # === TOPIC KEYWORD MATCHING ===
        stop_words = {'the', 'a', 'an', 'is', 'are', 'in', 'of', 'to', 'for',
                      'on', 'with', 'that', 'this', 'be', 'as', 'by', 'or', 'and',
                      'use', 'using', 'used', 'reports', 'error', 'errors', 'incorrect'}
        
        topic_words = [w for w in re.split(r'\W+', row_data.topic.lower()) 
                       if w and w not in stop_words and len(w) > 2]
        
        topic_weight = 1.5
        for word in topic_words[:6]:
            total_weight += topic_weight
            if word in page_text_lower:
                earned_weight += topic_weight
                scoring_details['topic_matches'][word] = True
            else:
                scoring_details['topic_matches'][word] = False
        
        # === TITLE BONUSES ===
        if framework in page_title_lower:
            bonus = 0.20 * total_weight
            earned_weight += bonus
            scoring_details['title_bonuses'].append({'type': 'framework_in_title', 'bonus': 0.20})
        
        title_anno_count = 0
        for anno in annotations[:5]:
            if anno.lower() in page_title_lower or anno.lstrip('@').lower() in page_title_lower:
                if title_anno_count < 2:
                    bonus = 0.15 * total_weight
                    earned_weight += bonus
                    scoring_details['title_bonuses'].append({'type': 'annotation_in_title', 'annotation': anno})
                    title_anno_count += 1
        
        # === FINAL SCORE ===
        final_score = min(earned_weight / max(total_weight, 1), 1.0)
        
        scoring_details['summary'] = {
            'total_weight': round(total_weight, 2),
            'earned_weight': round(earned_weight, 2),
            'final_score': round(final_score, 3),
            'annotations_checked': len(annotations),
            'annotations_found': sum(1 for a in scoring_details['annotation_matches'].values() if a['found'])
        }
        
        return final_score, scoring_details
    
    # =========================================================================
    # LLM CONTENT VALIDATION
    # =========================================================================
    
    def _llm_validate_content(
        self,
        url: str,
        content_snippet: str,
        page_title: str,
        framework: str,
        topic: str,
        annotations: List[str]
    ) -> dict:
        """Use GPT-(5) to validate if page content is relevant."""
        annotations_str = ', '.join(annotations) if annotations else "None found"
        
        prompt = f"""Analyze if this web page is relevant to the Java framework issue.

=== ISSUE ===
Framework: {framework}
Topic: {topic}
Required Annotations: {annotations_str}

=== WEB PAGE ===
URL: {url}
Title: {page_title}

Content:
{content_snippet[:1200]}

=== CRITERIA ===
1. Does it discuss {framework}?
2. Does it mention annotations: {annotations_str}?
3. Is it technical documentation or a tutorial?
4. Is it relevant to "{topic}"?

Respond with JSON only:
{{"is_relevant": true/false, "confidence": "high/medium/low", "reason": "one sentence", "annotations_discussed": ["@Anno1"]}}"""

        # try:
        #     response = self.client.chat.completions.create(
        #         model=MODEL_RULE_GENERATION,
        #         messages=[
        #             {"role": "system", "content": "Validate web page relevance. JSON only."},
        #             {"role": "user", "content": prompt}
        #         ],
        #         temperature=TEMPERATURE_GPT4,
        #         #max_tokens=200
        #     )

        try:
            response = self.client.responses.create(
                model="gpt-5",
                input=[{"role": "system", "content": "Validate web page relevance. JSON only."}, 
                    {"role": "developer", "content": prompt}],
                reasoning={"effort": "high"},
                text={"verbosity": "low"}
            )

            result_text = response.output_text.strip()
            result_text = re.sub(r'^```json\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)
            
            return json.loads(result_text)
            
        except Exception as e:

            return {"is_relevant": False, "confidence": "low", "reason": f"Error: {str(e)[:50]}"}
    
    # =========================================================================
    # URL VALIDATION WITH CONTENT RELEVANCE
    # =========================================================================
    
    def _open_3rd_party_found_URLs(
        self, 
        urls_text: str, 
        row_data: RowData,
        annotation_data: JetBrainsData,
        max_urls: int = URL_VALIDATION_MAX_URLS,
        min_relevance: float = URL_VALIDATION_MIN_RELEVANCE,
        timeout: int = URL_VALIDATION_TIMEOUT,
        use_llm_validation: bool = URL_VALIDATION_USE_LLM
    ) -> Tuple[str, List[dict]]:
        """
        Validate 3rd-party URLs using content relevance scoring + LLM verification.
        
        Args:
            urls_text: Text containing URLs from web search
            row_data: RowData with framework/topic info
            annotation_data: JetBrainsData with @annotations from JSON
            max_urls: Maximum validated URLs to return
            min_relevance: Minimum relevance score threshold (0.0-1.0)
            timeout: HTTP request timeout
            use_llm_validation: Use GPT-4 for borderline cases
        
        Returns:
            Tuple of (validated_urls_text, validation_report)
        """
        if not WEB_SCRAPING_AVAILABLE:
            return urls_text, [{'error': 'requests/beautifulsoup4 not available'}]
        
        # Extract URLs
        url_pattern = r'https?://[^\s\)\]\}\>\"\'\,]+'
        urls = re.findall(url_pattern, urls_text)
        
        validated_urls = []
        validation_report = []
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        print(f"      Validating {len(urls)} URLs ({annotation_data.annotation_count} annotations)...")
        
        for url in urls:
            url = re.sub(r'[.,;:!?\)\]\}]+$', '', url)
            
            # Skip excluded domains
            if any(domain in url.lower() for domain in self.EXCLUDED_DOMAINS):
                validation_report.append({
                    'url': url, 'status': 'skipped', 'reason': 'JetBrains domain'
                })
                continue
            
            try:
                response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                status_code = response.status_code
                final_url = response.url
                
                if status_code >= 400:
                    validation_report.append({
                        'url': url, 'status_code': status_code, 'is_valid': False,
                        'error': f'HTTP {status_code}'
                    })
                    continue
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                
                page_text = soup.get_text(separator=' ', strip=True)
                page_title = soup.title.string.strip() if soup.title and soup.title.string else ""
                
                # Calculate relevance score
                relevance_score, score_details = self._calculate_relevance_score(
                    page_text, page_title, row_data, annotation_data
                )
                
                # LLM validation for borderline cases - - - - - - - - > Should use a tool call approach to use an LLM model
                # to validate the URL
                llm_result = None
                if use_llm_validation and 0.25 <= relevance_score <= 0.55:
                    print(f"        LLM validating: {url[:50]}...")
                    llm_result = self._llm_validate_content(
                        final_url, page_text[:1500], page_title,
                        row_data.framework, row_data.topic,
                        annotation_data.annotations.get('all', [])
                    )
                    if llm_result.get('is_relevant', False):
                        relevance_score = max(relevance_score, 0.60)
                    else:
                        relevance_score = min(relevance_score, 0.25)
                
                is_valid = relevance_score >= min_relevance
                
                # Create validation report entry
                url_report = {
                    'url': url,
                    'final_url': final_url,
                    'status_code': status_code,
                    'page_title': page_title[:100] if page_title else None,
                    'is_valid': is_valid,
                    'relevance_score': round(relevance_score, 3),
                    'annotations_found': score_details['summary']['annotations_found'],
                    'llm_validation': llm_result
                }
                
                validation_report.append(url_report)
                
                
                if is_valid:
                    validated_urls.append({
                        'url': final_url,
                        'title': page_title[:100] if page_title else "No title",
                        'relevance_score': relevance_score,
                        'annotations_found': score_details['summary']['annotations_found']
                    })
            
            except requests.exceptions.Timeout:
                validation_report.append({'url': url, 'is_valid': False, 'error': 'Timeout'})
            except Exception as e:
                validation_report.append({'url': url, 'is_valid': False, 'error': str(e)[:100]})
        
        # Sort by relevance and take top N
        validated_urls.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Deduplicate
        unique_urls = []
        seen = set()
        for item in validated_urls:
            if item['url'] not in seen:
                seen.add(item['url'])
                unique_urls.append(item)
        
        top_urls = unique_urls[:max_urls]
        
        if top_urls:
            lines = []
            for i, v in enumerate(top_urls, 1):
                anno_info = f"[{v['annotations_found']}/{annotation_data.annotation_count} annos, {v['relevance_score']:.0%}]"
                lines.append(f"{i}. {v['url']} - {v['title']} {anno_info}")
            validated_text = "\n".join(lines)
        else:
            validated_text = "No relevant 3rd-party URLs could be verified."
        
        print(f"      Result: {len(validated_urls)} passed, {len(urls) - len(validated_urls)} failed")
        
        # Save all URL reports for this row to a single JSON file
        if validation_report:
            try:
                from datetime import datetime
                
                # Create filename based on framework and topic
                safe_framework = re.sub(r'[^\w\-]', '_', row_data.framework.lower())[:30]
                safe_topic = re.sub(r'[^\w\-]', '_', row_data.topic.lower())[:50]
                filename = f"url_report_{safe_framework}_{safe_topic}.json"
                
                # Create consolidated report with metadata
                consolidated_report = {
                    '_metadata': {
                        'saved_at': datetime.now().isoformat(),
                        'report_file': filename,
                        'framework': row_data.framework,
                        'topic': row_data.topic,
                        'total_urls_checked': len(urls),
                        'urls_passed': len(validated_urls),
                        'urls_failed': len(urls) - len(validated_urls)
                    },
                    'url_validations': validation_report
                }
                
                # Save to file
                report_path = os.path.join(BASE_DIR, "url_reports", filename)
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(consolidated_report, f, indent=2, ensure_ascii=False)
                
                print(f"      Saved URL report: {filename}")
                
            except Exception as e:
                print(f"      Warning: Could not save consolidated URL report: {str(e)[:50]}")
        
        return validated_text, validation_report
    
    # =========================================================================
    # WEB SEARCH USING GPT-5 WITH WEB_SEARCH TOOL
    # =========================================================================
    
    def search_relevant_web_pages(self, row_data: RowData, row_index: int = 0) -> Tuple[str, JetBrainsData]:
        """Search for relevant 3rd-party web pages using GPT-5 with web_search"""
        search_candidates, annotation_data = self._build_search_candidates(row_data, row_index)
        
        search_prompt = f"""Search the web for the most relevant documentation pages for this Java framework issue.

=== ISSUE ===
Framework: {row_data.framework}
Topic: {row_data.topic}
Description: {row_data.description}
Annotations to find: {', '.join(annotation_data.annotations.get('all', [])[:5])}

=== SEARCH QUERIES ===
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(search_candidates[:10]))}

=== REQUIREMENTS ===
1. Find 3-5 relevant pages (we will validate and select top {URL_VALIDATION_MAX_URLS})
2. EXCLUDE all JetBrains URLs
3. PRIORITIZE: Official docs > Baeldung > Stack Overflow

Return URLs with brief descriptions."""

        # GPT-4 model used this format for calling web sesarch feature of OpenAI via API
        # try:
        #     response = self.client.responses.create(
        #         model="gpt-5",
        #         reasoning={"effort": WEB_SEARCH_REASONING_EFFORT},
        #         tools=[{"type": "web_search", "search_context_size": WEB_SEARCH_CONTEXT_SIZE}],
        #         tool_choice="auto",
        #         input=search_prompt
        #     )
        
        
        try:
            response = self.client.responses.create(
                model=MODEL_WEB_SEARCH,
                tools=[{"type": "web_search"}],
                input=search_prompt,
             
            )
            
            result = self._filter_jetbrains_urls(response.output_text) if response.output_text else ""
            return result, annotation_data
                
        except Exception as e:
            print("  Failed with GPT-5.2 Search!! ..............FALLBACK to 3 Constructed URLs")
            fallback = self._construct_fallback_urls(row_data)
            return fallback, annotation_data
    
    # def _fallback_web_search(self, row_data: RowData, candidates: List[str]) -> str:
    #     """Fallback web search using gpt-4o-search-preview"""
    #     collected = []
    #     for query in candidates[:5]:
    #         try:
    #             response = self.client.chat.completions.create(
    #                 model="gpt-4o-search-preview",
    #                 web_search_options={"search_context_size": "high"},
    #                 messages=[{"role": "user", "content": f"Search: {query}\nFind docs (exclude JetBrains)."}]
    #             )
    #             urls = re.findall(r'https?://[^\s\)\]\}\>\"\'\,]+', response.choices[0].message.content)
    #             for url in urls:
    #                 url = re.sub(r'[.,;:!?\)\]\}]+$', '', url)
    #                 if not any(d in url.lower() for d in self.EXCLUDED_DOMAINS):
    #                     if url not in collected:
    #                         collected.append(url)
    #             if len(collected) >= 5:
    #                 break
    #         except:
    #             continue
        
    #     if collected:
    #         return '\n'.join(f"{i+1}. {u}" for i, u in enumerate(collected[:5]))
    #     return self._construct_fallback_urls(row_data)
    
    def _construct_fallback_urls(self, row_data: RowData) -> str:
        """Construct fallback URLs from FRAMEWORK_DOCS"""
        framework_lower = row_data.framework.lower() if row_data.framework else 'java'
        
        for key, docs in FALLBACK_WEBPAGES.items():
            if key in framework_lower or framework_lower in key:
                lines = [f"{i+1}. {url} - {desc}" for i, (url, desc) in enumerate(docs[:2])]
                return '\n'.join(lines)
        
        return f"[FBO] 1. https://docs.oracle.com/javaee/7/tutorial/ - Java EE Tutorial\n2. https://stackoverflow.com/questions/tagged/{framework_lower.replace(' ', '-')}"

   
   
    def _filter_jetbrains_urls(self, text: str) -> str:
        lines = text.split('\n')
        return '\n'.join(l for l in lines if not any(d in l.lower() for d in self.EXCLUDED_DOMAINS))
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    
   
    def report_none_existing_functions(self, row_data: RowData, generated_rule: str, 
                    rule_description: str, web_pages: str) -> str:
        """Evaluate the rule using GPT-5 with tool call approach."""
        builtin_functions_list = self._format_builtins_list()
        
 
        context = f"""Evaluate this RSL rule:

    === INSPECTION DATA ===
    Framework: {row_data.framework}
    Topic: {row_data.topic}
    Issue: {row_data.description}

    === GENERATED RSL RULE ===
    {generated_rule}

    === RULE DESCRIPTION ===
    {rule_description}

    === FALLBACK WEB RESOURCES ===
    {web_pages}

    === EXISING FUNCTION LIST ===
    {builtin_functions_list}
    
    
    From the generated rule, list any non-existing function name."""

        instructions = """You are an expert in metadata-related bugs in Java applications. 
    Validate the GPT-5-generated rule based on the framework, topic, issue description, and the existing builtin function list. 
   Check if the rule uses only these valid functions. Report any non-existing functions following these steps:
    1. Extract the function names used in the generated rule.
    2. Check if a function name in the generated rule is seen in the builtin_functions list.
    3. If a function name in the generated rule not seen in the builtin_functions list, tell me the function name.
    """

        messages = [
            {"role": "developer", "content": context}
        ]
        result = ""
        result1 = ""
        try:
            # Simpler call without tool forcing for fallback
            print("  STARTING with GPT-5 Eval!!...")
            resp = self.client.responses.create(
                model=MODEL_EVALUATION,
                input=messages,
                instructions=instructions,
                reasoning={"effort": "high"},
                text={"verbosity": "low"}
            )

            result = resp.output_text.strip() if hasattr(resp, 'output_text') else "No - Evaluation failed"
            
            # Sanitize for CSV/Excel - prevent #NAME! errors
            if result and result[0] in ['=', '-', '+', '@']:
                result = "'" + result  # Prefix with single quote to force text interpretation
            
            return result
        except Exception as e:
            return self._fallback_evaluation(row_data, generated_rule, rule_description)

    
    def evaluate_rule(self, row_data: RowData, generated_rule: str, 
                    rule_description: str, web_pages: str) -> str:
        """Evaluate the rule using GPT-5 with tool call approach."""
        builtin_functions_list = self._format_builtins_list()
        
        context = f"""Evaluate this RSL rule:

    === INSPECTION DATA ===
    Framework: {row_data.framework}
    Topic: {row_data.topic}
    Issue: {row_data.description}

    === GENERATED RSL RULE ===
    {generated_rule}

    === RULE DESCRIPTION ===
    {rule_description}

    === FALLBACK WEB RESOURCES ===
    {web_pages}

    === EXISING FUNCTION LIST ===
    {builtin_functions_list}
    
    
    From the generated rule, list any non-existing function name."""

        instructions = """You are an expert in metadata-related bugs in Java applications. 
                Validate the GPT-5-generated rule based on the framework, topic, issue description, and the given example. 
                If the rule is correct, submit 'Yes.' as the result. If the rule is incorrect, submit 'No' with a brief explanation.
                1. Your validation result should not be influenced  by the none-existing functions in the rule if any. 
                2  Indicate "Yes" or "No" first in terms of the generated rule's correctness with respect to the given example. 
                3. If your answer is "No", briefly explain why the rule is incorrect."""

        messages = [
                {"role": "developer", "content": context}
            ]
        result = ""
        
        try:
            # Simpler call without tool forcing for fallback
            print("  STARTING with GPT-5 Eval!!...")
            resp = self.client.responses.create(
                model=MODEL_EVALUATION,
                input=messages,
                instructions=instructions,
                reasoning={"effort": "high"},
                text={"verbosity": "low"}
            )

            result = resp.output_text.strip() if hasattr(resp, 'output_text') else "No - Evaluation failed"
            
            # # Sanitize for CSV/Excel - prevent #NAME! errors
            # if result and result[0] in ['=', '-', '+', '@']:
            #     result = "'" + result  # Prefix with single quote to force text interpretation
            
            return result
        except Exception as e:
            return self._fallback_evaluation(row_data, generated_rule, rule_description)

           
   


    def _fallback_evaluation(self, row_data: RowData, generated_rule: str, 
                            rule_description: str) -> str:
        """Fallback evaluation using constructed URLs when primary evaluation fails."""
        
        # Get fallback web pages and builtin functions list
        fallback_web_pages = self._construct_fallback_urls(row_data)
        builtin_functions_list = self._format_builtins_list()
        
        # --- Build fallback context ---
        context = f"""Evaluate this RSL rule:

    === INSPECTION DATA ===
    Framework: {row_data.framework}
    Topic: {row_data.topic}
    Issue: {row_data.description}

    === GENERATED RSL RULE ===
    {generated_rule}

    === RULE DESCRIPTION ===
    {rule_description}

    === FALLBACK WEB RESOURCES ===
    {fallback_web_pages}

    === {builtin_functions_list} ===
    Check if the rule uses only these valid functions. Report any none-existing functions following these steps:
    1. Extract the function names with parentheses, (), from the generated rule.
    2. Check if the function name is seen in the builtin_functions list.
    3. If the function name is not seen, report it.
    

    From each generated rule, simply list all none-existing functions that satisfy the statements above."""

        instructions = """You are an expert in metadata-related bugs in Java applications. 
    Validate the GPT-5-generated rule based on the topic, content, and the given example. 
    If the rule is correct, submit 'Yes.' as the result. If the rule is incorrect, submit 'No' with a brief explanation.
    Your evaluation should not consider if there are any non-existing functions in the rule yet.
    To answer "Correct," the generated rule should be coherent with the topic, issue description, 
    rule description and located web pages's content. Your validation result should not be influenced
    by the non-existing functions in the rule. Indicate "Yes" or "No" first, then report 
    any none-existing functions name: example_new_function_name()."""

        messages = [
            {"role": "developer", "content": context}
        ]

        try:
            # Simpler call without tool forcing for fallback
            print("  Failed with GPT-5 Eval!!  [4/5] Fallback evaluation (GPT-5)...")
            resp = self.client.responses.create(
                model=MODEL_EVALUATION,
                input=messages,
                instructions=instructions,
                reasoning={"effort": "high"},
                text={"verbosity": "low"}
            )
            
            result = resp.output_text.strip() if hasattr(resp, 'output_text') else "No - Evaluation failed"
            return f"[FBO] {result}"
            
        except Exception as e:
            # Ultimate fallback - return a safe default with FBO indicator
            return f"[FBO] No - Fallback evaluation failed: {str(e)}"


    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================

    def process_row(self, row_data: RowData, row_index: int = 0) -> GeneratedContent:
        """Process a single row through the complete pipeline"""
        print(f"  Processing: {row_data.topic[:60]}...")
        
        print("    [1/5] Generating RSL-Expressed rule (GPT-5)...")
        rule = self.generate_rule(row_data)
        time.sleep(0.5)
        
        print("    [2/5] Generating Desccription (GPT-5)...")
        rule_description = self.generate_rule_description(row_data, rule)
        time.sleep(0.5)
        
        print("    [3/5] Web Search (GPT-5)...")
        web_pages_raw, annotation_data = self.search_relevant_web_pages(row_data, row_index)
        time.sleep(0.5)
        
        print("    [4/5] Validating 3rd-party URLs ...")
        web_pages, validation_report = self._open_3rd_party_found_URLs(
            web_pages_raw, row_data, annotation_data
        )
        time.sleep(0.5)
        
        print("    [5/5] Report Non-existing functions in the generated rule (GPT-5)...")
        none_existing_functions = self.report_none_existing_functions(row_data, rule, rule_description, web_pages)

        print("    [5/5-1] Evaluating the RULE (GPT-5)...")
        evaluation = self.evaluate_rule(row_data, rule, rule_description, web_pages)
        
        return GeneratedContent(rule, rule_description, web_pages, none_existing_functions, evaluation)

    def save_annotations_json(self):
        """Save extracted annotations to JSON"""
        return self.ruleMakerJB.save_to_json()


# =============================================================================
# CSV UTILITIES
# =============================================================================

def read_csv_file(file_path: str) -> List[Dict[str, str]]:
    rows = []
    for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            with open(file_path, 'r', encoding=encoding, newline='') as f:
                rows = list(csv.DictReader(f))
            print(f"Read CSV with {encoding} encoding")
            break
        except:
            continue
    return rows


def write_csv_file(file_path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# MAIN
# =============================================================================

"""
MODIFIED MAIN FUNCTION WITH:
1. Periodic saves every 5th row
2. Graceful shutdown on keyboard interrupt (Ctrl+C)
3. Graceful shutdown on network errors
4. Emergency save on any exception

Replace your existing main() function with this version.
"""

import signal
import sys
import atexit

# =============================================================================
# GLOBAL STATE FOR EMERGENCY SAVES
# =============================================================================

# Global variables to track state for emergency saves
_emergency_save_state = {
    'generator': None,
    'output_rows': [],
    'output_columns': [],
    'last_saved_count': 0
}


def emergency_save():
    """Emergency save function called on abrupt termination."""
    state = _emergency_save_state
    
    if not state['output_rows']:
        print("\n[Emergency Save] No data to save.")
        return
    
    print(f"\n[Emergency Save] Saving {len(state['output_rows'])} rows...")
    
    try:
        # Save CSV
        if state['output_columns']:
            emergency_csv_path = OUTPUT_CSV_PATH.replace('.csv', '_emergency.csv')
            write_csv_file(emergency_csv_path, state['output_rows'], state['output_columns'])
            print(f"  ✓ Saved CSV: {emergency_csv_path}")
        
        # Save annotations JSON
        if state['generator'] and hasattr(state['generator'], 'save_annotations_json'):
            state['generator'].save_annotations_json()
            print(f"  ✓ Saved annotations JSON")
        
        print("[Emergency Save] Complete!")
        
    except Exception as e:
        print(f"[Emergency Save] ERROR: {e}")


def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals."""
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\n\n{'='*60}")
    print(f"INTERRUPTED! Received signal: {signal_name}")
    print(f"{'='*60}")
    emergency_save()
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination request

# Register emergency save on normal exit too
atexit.register(emergency_save)


# =============================================================================
# PERIODIC SAVE FUNCTION
# =============================================================================

def periodic_save(output_rows: List[Dict], output_columns: List[str], 
                  generator, count: int, force: bool = False):
    """
    Save progress every 5 rows or when forced.
    
    Args:
        output_rows: Current list of processed rows
        output_columns: Column names for CSV
        generator: JetBrainsRuleGenerator instance
        count: Current row count
        force: Force save regardless of count
    """
    global _emergency_save_state
    
    # Update global state for emergency saves
    _emergency_save_state['generator'] = generator
    _emergency_save_state['output_rows'] = output_rows
    _emergency_save_state['output_columns'] = output_columns
    
    # Save every 5 rows or when forced
    if force or (count > 0 and count % 5 == 0):
        print(f"\n  [Periodic Save] Saving progress at row {count}...")
        
        try:
            # Save CSV with timestamp suffix for safety
            timestamp = datetime.now().strftime("%H%M%S")
            periodic_csv_path = OUTPUT_CSV_PATH.replace('.csv', f'_progress_{count}.csv')
            write_csv_file(periodic_csv_path, output_rows, output_columns)
            print(f"    ✓ CSV saved: {periodic_csv_path}")
            
            # Also update the main output file
            write_csv_file(OUTPUT_CSV_PATH, output_rows, output_columns)
            print(f"    ✓ Main CSV updated: {OUTPUT_CSV_PATH}")
            
            # Save annotations JSON
            if generator:
                generator.save_annotations_json()
                print(f"    ✓ Annotations JSON saved")
            
            _emergency_save_state['last_saved_count'] = count
            print(f"  [Periodic Save] Complete!")
            
        except Exception as e:
            print(f"  [Periodic Save] WARNING: {e}")


# =============================================================================
# MODIFIED MAIN FUNCTION
# =============================================================================

def main():
    global _emergency_save_state
    
    print("=" * 80)
    print("JetBrains Inspection RSL Rule Generator")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Models: GPT-5 (rule gen and desc.), GPT-5 (web search), GPT-5 (eval)")
    print(f"  URL Validation: max={URL_VALIDATION_MAX_URLS}, min_relevance={URL_VALIDATION_MIN_RELEVANCE}, LLM={URL_VALIDATION_USE_LLM}")
    print(f"  Frameworks: {len(FALLBACK_WEBPAGES)} configured")
    print(f"  Auto-save: Every 5 rows")
    print(f"  Emergency save: On Ctrl+C or errors")
    print("=" * 80)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        try:
            key_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "YOUR_KEY_IN_ENV_DONT_PUBLIC.txt")
            if os.path.exists(key_file_path):
                with open(key_file_path, "r") as f:
                    api_key = f.read().strip()
                print("Loaded API key from YOUR_KEY_IN_ENV_DONT_PUBLIC.txt")
        except Exception as e:
            print(f"Failed to load key from file: {e}")

    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not set - running test mode")
        return
    
    generator = None
    output_rows = []
    output_columns = []
    count = 0
    
    try:
        generator = JetBrainsRuleGenerator(api_key)
        input_rows = read_csv_file(INPUT_CSV_PATH)
        print(f"\nProcessing {len(input_rows)} rows...")
        
        original_columns = list(input_rows[0].keys())
        new_columns = [
            "GPT-5 Generated Rule", 
            "GPT-5 Description of Rule",
            "3rd-Party Web Pages (No-JetBrains)", 
            "GPT-5-reported None-Existing Functions", 
            "GPT-5 Evaluation Opinion"
        ]
        output_columns = original_columns + new_columns
        
        # Update global state immediately
        _emergency_save_state['generator'] = generator
        _emergency_save_state['output_columns'] = output_columns
        
        for idx, row in enumerate(input_rows):
            if MAX_ROWS_TO_PROCESS and count >= MAX_ROWS_TO_PROCESS:
                break
            
            topic = row.get('TOPIC from JetBrains', '').strip()
            if not topic:
                output_row = dict(row)
                for col in new_columns:
                    output_row[col] = ""
                output_rows.append(output_row)
                continue
            
            count += 1
            print(f"\n[{count}] Row {idx + 1}")
            
            try:
                row_data = RowData(
                    framework=row.get('Name of FRAMEWORK', ''),
                    source=row.get('SOURCE from JetBrains', ''),
                    topic=topic,
                    description=row.get('REPORTED ISSUE DESCRIPTIONS from JetBrains', ''),
                    examples=row.get('EXAMPLE(S) from SOURCE', '')
                )
                
                generated = generator.process_row(row_data, idx)
                
                output_row = dict(row)
                output_row["GPT-5 Generated Rule"] = generated.rule
                output_row["GPT-5 Description of Rule"] = generated.rule_description
                output_row["3rd-Party Web Pages (No-JetBrains)"] = generated.web_pages
                output_row["GPT-5-reported None-Existing Functions"] = generated.none_existing_functions
                output_row["GPT-5 Evaluation Opinion"] = generated.evaluation
                output_rows.append(output_row)
                
                # =============================================
                # PERIODIC SAVE: Every 5th row
                # =============================================
                periodic_save(output_rows, output_columns, generator, count)
                
                time.sleep(DELAY_BETWEEN_ROWS)
                
            except KeyboardInterrupt:
                # Re-raise to be caught by outer handler
                raise
                
            except (ConnectionError, TimeoutError, OSError) as e:
                # Network-related errors - save and continue or abort
                print(f"\n⚠️  Network error on row {count}: {e}")
                print("Saving progress before potential retry...")
                periodic_save(output_rows, output_columns, generator, count, force=True)
                
                # Ask user whether to continue
                try:
                    response = input("Continue processing? (y/n): ").strip().lower()
                    if response != 'y':
                        raise KeyboardInterrupt("User chose to stop after network error")
                except EOFError:
                    # Non-interactive mode - save and exit
                    raise
                    
            except Exception as e:
                # Other errors - log and continue
                print(f"\n⚠️  Error processing row {count}: {e}")
                output_row = dict(row)
                output_row["GPT-5 Generated Rule"] = f"ERROR: {str(e)}"
                output_row["GPT-5 Description of Rule"] = ""
                output_row["3rd-Party Web Pages (No-JetBrains)"] = ""
                output_row["GPT-5-reported None-Existing Functions"] = ""
                output_row["GPT-5 Evaluation Opinion"] = ""
                output_rows.append(output_row)
                
                # Still do periodic save
                periodic_save(output_rows, output_columns, generator, count)
        
        # =============================================
        # FINAL SAVE: After all rows processed
        # =============================================
        print(f"\n\n{'=' * 80}")
        print("FINAL SAVE")
        print(f"{'=' * 80}")
        
        # Save annotations JSON
        generator.save_annotations_json()
        
        # Write final output CSV
        write_csv_file(OUTPUT_CSV_PATH, output_rows, output_columns)
        
        # Clear emergency state since we saved successfully
        _emergency_save_state['output_rows'] = []
        
        print(f"\n{'=' * 80}")
        print("COMPLETE!")
        print(f"  Processed: {count} rows")
        print(f"  Output CSV: {OUTPUT_CSV_PATH}")
        print(f"  Annotations JSON: {ANNOTATIONS_JSON_PATH}")
        print(f"{'=' * 80}")
        
    except KeyboardInterrupt:
        print(f"\n\n{'=' * 80}")
        print("INTERRUPTED BY USER (Ctrl+C)")
        print(f"{'=' * 80}")
        print(f"Processed {count} rows before interruption.")
        
        # Save what we have
        if output_rows and output_columns:
            print("Saving progress...")
            interrupted_csv_path = OUTPUT_CSV_PATH.replace('.csv', '_interrupted.csv')
            write_csv_file(interrupted_csv_path, output_rows, output_columns)
            print(f"  ✓ Saved: {interrupted_csv_path}")
            
            # Also update main file
            write_csv_file(OUTPUT_CSV_PATH, output_rows, output_columns)
            print(f"  ✓ Updated: {OUTPUT_CSV_PATH}")
            
            if generator:
                generator.save_annotations_json()
                print(f"  ✓ Saved annotations JSON")
        
        # Clear emergency state
        _emergency_save_state['output_rows'] = []
        print("Goodbye!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"❌ FATAL ERROR: {e}")
        print(f"{'=' * 80}")
        
        import traceback
        traceback.print_exc()
        
        # Emergency save
        if output_rows and output_columns:
            print("\nAttempting emergency save...")
            error_csv_path = OUTPUT_CSV_PATH.replace('.csv', '_error.csv')
            try:
                write_csv_file(error_csv_path, output_rows, output_columns)
                print(f"  ✓ Saved: {error_csv_path}")
                
                if generator:
                    generator.save_annotations_json()
                    print(f"  ✓ Saved annotations JSON")
                    
            except Exception as save_error:
                print(f"  ❌ Emergency save failed: {save_error}")
        
        # Clear emergency state
        _emergency_save_state['output_rows'] = []
        sys.exit(1)


if __name__ == "__main__":
    main()




# def main():
#     print("=" * 80)
#     print("JetBrains Inspection RSL Rule Generator")
#     print("=" * 80)
#     print(f"\nConfiguration:")
#     print(f"  Models: GPT-5 (rule gen and desc.), GPT-5 (web search), GPT-5 (eval)")
#     print(f"  URL Validation: max={URL_VALIDATION_MAX_URLS}, min_relevance={URL_VALIDATION_MIN_RELEVANCE}, LLM={URL_VALIDATION_USE_LLM}")
#     print(f"  Frameworks: {len(FALLBACK_WEBPAGES)} configured")
#     print("=" * 80)
    
#     api_key = os.getenv("OPENAI_API_KEY")
    
#     if not api_key:
#         try:
#              # Try reading from file
#             key_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "YOUR_KEY_IN_ENV_DONT_PUBLIC.txt")
#             if os.path.exists(key_file_path):
#                 with open(key_file_path, "r") as f:
#                     api_key = f.read().strip()
#                 print("Loaded API key from YOUR_KEY_IN_ENV_DONT_PUBLIC.txt")
#         except Exception as e:
#             print(f"Failed to load key from file: {e}")

#     if not api_key:
#         print("\n⚠️  OPENAI_API_KEY not set - running test mode")
#         #generate_sample_output()
#         return
    
#     try:
#         generator = JetBrainsRuleGenerator(api_key)
#         input_rows = read_csv_file(INPUT_CSV_PATH)
#         print(f"\nProcessing {len(input_rows)} rows...")
        
#         original_columns = list(input_rows[0].keys())
#         new_columns = ["GPT-5 Generated Rule", "GPT-5 Description of Rule",
#                        "3rd-Party Web Pages (No-JetBrains)", "GPT-5-reported None-Existing Functions", "GPT-5 Evaluation Opinion"]
#         output_columns = original_columns + new_columns
        
#         output_rows = []
#         count = 0
        
#         for idx, row in enumerate(input_rows):
#             if MAX_ROWS_TO_PROCESS and count >= MAX_ROWS_TO_PROCESS:
#                 break
            
#             topic = row.get('TOPIC from JetBrains', '').strip()
#             if not topic:
#                 output_row = dict(row)
#                 for col in new_columns:
#                     output_row[col] = ""
#                 output_rows.append(output_row)
#                 continue
            
#             count += 1
#             print(f"\n[{count}] Row {idx + 1}")
            
#             row_data = RowData(
#                 framework=row.get('Name of FRAMEWORK', ''),
#                 source=row.get('SOURCE from JetBrains', ''),
#                 topic=topic,
#                 description=row.get('REPORTED ISSUE DESCRIPTIONS from JetBrains', ''),
#                 examples=row.get('EXAMPLE(S) from SOURCE', '')
#             )
            
#             generated = generator.process_row(row_data, idx)
            
#             output_row = dict(row)
#             output_row["GPT-5 Generated Rule"] = generated.rule
#             output_row["GPT-5 Description of Rule"] = generated.rule_description
#             output_row["3rd-Party Web Pages (No-JetBrains)"] = generated.web_pages
#             output_row["GPT-5-reported None-Existing Functions"] = generated.none_existing_functions
#             output_row["GPT-5 Evaluation Opinion"] = generated.evaluation
#             output_rows.append(output_row)
            
#             time.sleep(DELAY_BETWEEN_ROWS)
            
#         # Save annotations JSON
#         generator.save_annotations_json()
        
#         # Write output CSV
#         write_csv_file(OUTPUT_CSV_PATH, output_rows, output_columns)
        
#         print(f"\n\n{'=' * 80}")
#         print("COMPLETE!")
#         print(f"  Output CSV: {OUTPUT_CSV_PATH}")
#         print(f"  Annotations JSON: {ANNOTATIONS_JSON_PATH}")
#         print(f"{'=' * 80}")
        
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
      


# if __name__ == "__main__":
#     main()
