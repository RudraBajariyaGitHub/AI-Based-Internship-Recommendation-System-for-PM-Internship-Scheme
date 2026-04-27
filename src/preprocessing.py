import re
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from fuzzywuzzy import process, fuzz
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MLPreprocessor:

    def __init__(self):
        self.missing_vals = {
            'nan', 'none', 'null', 'n/a', 'na', 'n.a.', 'undefined',
            'unknown', 'missing', 'blank', 'empty', '', '-', '--',
            '?', '??', 'no data', 'no value', 'not available',
            'not provided', 'tbd', 'todo', 'xxx', 'N/A', 'None', 'NIL'
        }

        self.branch_map = {
            'computer science': 'Computer Science',
            'cse': 'Computer Science',
            'cs': 'Computer Science',
            'information technology': 'IT',
            'it': 'IT',
            'data science': 'Data Science',
            'ds': 'Data Science',
            'artificial intelligence': 'AI',
            'ai': 'AI',
            'ai & ml': 'AI',
            'ai/ml': 'AI',
            'machine learning': 'AI',
            'electronics & communication': 'Electronics & Communication',
            'electronics and communication': 'Electronics & Communication',
            'ece': 'Electronics & Communication',
            'electronics': 'Electronics & Communication',
            'electrical engineering': 'Electrical Engineering',
            'electrical engg': 'Electrical Engineering',
            'ee': 'Electrical Engineering',
            'civil engineering': 'Civil Engineering',
            'civil': 'Civil Engineering',
            'ce': 'Civil Engineering',
            'mechanical engineering': 'Mechanical Engineering',
            'mechanical': 'Mechanical Engineering',
            'me': 'Mechanical Engineering',
            'chemical engineering': 'Chemical Engineering',
            'chemical engg': 'Chemical Engineering',
            'biotechnology': 'Biotechnology',
            'bio tech': 'Biotechnology',
            'architecture': 'Architecture',
            'finance': 'Finance',
            'economics': 'Finance'
        }

        self.tier_map = {
            'tier1': 'Tier-1', 'tier-1': 'Tier-1', 'tier 1': 'Tier-1', 'tier1': 'Tier-1',
            'tier2': 'Tier-2', 'tier-2': 'Tier-2', 'tier 2': 'Tier-2', 'tier2': 'Tier-2',
            'tier3': 'Tier-3', 'tier-3': 'Tier-3', 'tier 3': 'Tier-3', 'tier3': 'Tier-3'
        }

        self.work_type_map = {
            'remote': 'Remote',
            'wfh': 'Remote',
            'work from home': 'Remote',
            'work-from-home': 'Remote',
            'hybrid': 'Hybrid',
            'hybrid mode': 'Hybrid',
            'on-site': 'On-site',
            'onsite': 'On-site',
            'in-office': 'On-site',
            'office': 'On-site',
            'in office': 'On-site'
        }

        self.skill_standardization = {
            'python': 'Python', 'py': 'Python',
            'java': 'Java', 'javascript': 'JavaScript', 'js': 'JavaScript',
            'c++': 'C++', 'c#': 'C#', 'c sharp': 'C#',
            'html': 'HTML', 'css': 'CSS', 'sql': 'SQL',
            'react': 'React', 'reactjs': 'React', 'react.js': 'React',
            'angular': 'Angular', 'vue': 'Vue.js', 'node.js': 'Node.js', 'nodejs': 'Node.js',
            'express': 'Express.js', 'express.js': 'Express.js',
            'django': 'Django', 'flask': 'Flask', 'fastapi': 'FastAPI', 'fast api': 'FastAPI',
            'spring': 'Spring Boot', 'spring boot': 'Spring Boot',
            'dotnet': '.NET', 'asp.net': 'ASP.NET',

            'machine learning': 'Machine Learning', 'ml': 'Machine Learning',
            'deep learning': 'Deep Learning', 'dl': 'Deep Learning',
            'neural networks': 'Neural Networks',
            'tensorflow': 'TensorFlow', 'pytorch': 'PyTorch',
            'scikit-learn': 'Scikit-learn', 'sklearn': 'Scikit-learn',
            'keras': 'Keras', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
            'nlp': 'NLP', 'natural language processing': 'NLP',
            'computer vision': 'Computer Vision', 'cv': 'Computer Vision',
            'reinforcement learning': 'Reinforcement Learning',
            'time series': 'Time Series Analysis',
            'statistical analysis': 'Statistical Analysis',
            'data analysis': 'Data Analysis', 'data analytics': 'Data Analysis',
            'data visualization': 'Data Visualization',
            'tableau': 'Tableau', 'powerbi': 'Power BI', 'power bi': 'Power BI',
            'd3.js': 'D3.js', 'matplotlib': 'Matplotlib', 'seaborn': 'Seaborn',

            'mysql': 'MySQL', 'postgresql': 'PostgreSQL', 'postgres': 'PostgreSQL',
            'mongodb': 'MongoDB', 'mongo': 'MongoDB', 'redis': 'Redis', 'cassandra': 'Cassandra',
            'hadoop': 'Hadoop', 'spark': 'Apache Spark', 'apache spark': 'Apache Spark',
            'kafka': 'Apache Kafka', 'apache kafka': 'Apache Kafka',
            'hive': 'Apache Hive', 'pig': 'Apache Pig',

            'aws': 'AWS', 'amazon web services': 'AWS',
            'azure': 'Microsoft Azure', 'microsoft azure': 'Microsoft Azure',
            'gcp': 'Google Cloud Platform', 'google cloud': 'Google Cloud Platform',
            'google cloud platform': 'Google Cloud Platform',
            'docker': 'Docker', 'kubernetes': 'Kubernetes', 'k8s': 'Kubernetes',
            'jenkins': 'Jenkins', 'gitlab': 'GitLab', 'github': 'GitHub',
            'terraform': 'Terraform', 'ansible': 'Ansible',
            'linux': 'Linux', 'bash': 'Bash', 'shell scripting': 'Shell Scripting',

            'cybersecurity': 'Cybersecurity', 'cyber security': 'Cyber Security',
            'ethical hacking': 'Ethical Hacking',
            'network security': 'Network Security',
            'penetration testing': 'Penetration Testing',
            'security analysis': 'Security Analysis',
            'siem': 'SIEM',
            'blockchain': 'Blockchain', 'ethereum': 'Ethereum',
            'smart contracts': 'Smart Contracts', 'solidity': 'Solidity',
            'web3.js': 'Web3.js', 'web3': 'Web3.js',
            'cryptocurrency': 'Cryptocurrency',
            'iot': 'IoT', 'internet of things': 'IoT',
            'robotics': 'Robotics', 'ros': 'ROS',
            'matlab': 'MATLAB', 'simulink': 'Simulink',
            'autocad': 'AutoCAD', 'solidworks': 'SolidWorks',
            'ansys': 'ANSYS', 'staad.pro': 'STAAD.Pro', 'staad pro': 'STAAD.Pro',
            'aspen plus': 'Aspen Plus',
            'project management': 'Project Management',
            'excel': 'Excel', 'powerpoint': 'PowerPoint', 'word': 'Word',
            'communication': 'Communication Skills', 'leadership': 'Leadership',
            'problem solving': 'Problem Solving', 'critical thinking': 'Critical Thinking',

            'embedded systems': 'Embedded Systems',
            'arduino': 'Arduino', 'raspberry pi': 'Raspberry Pi',
            'sensors': 'Sensors', 'mqtt': 'MQTT',
            'circuit design': 'Circuit Design',
            'vhdl': 'VHDL',
            'gel electrophoresis': 'Gel Electrophoresis',
            'bioinformatics': 'Bioinformatics',
            'biotechnology': 'Biotechnology',
            'pcr': 'PCR',
            'figma': 'Figma',
            'ui design': 'UI Design', 'ux design': 'UX Design',
            'prototyping': 'Prototyping', 'wireframing': 'Wireframing',
            'adobe xd': 'Adobe XD', 'sketch': 'Sketch',
            'photoshop': 'Photoshop', 'illustrator': 'Illustrator',
            'design thinking': 'Design Thinking',
            'business analysis': 'Business Analysis',
            'financial modeling': 'Financial Modeling',
            'bloomberg': 'Bloomberg Terminal', 'bloomberg terminal': 'Bloomberg Terminal',
            'machne learning': 'Machine Learning',
            'fintech': 'Fintech',
            'devops': 'DevOps', 'dev ops': 'DevOps', 'dev-ops': 'DevOps',
            'ci/cd': 'CI/CD', 'cicd': 'CI/CD',
            'data engineering': 'Data Engineering',
            'rest api': 'REST API', 'api': 'API',
            'graphql': 'GraphQL',
            'microservices': 'Microservices',
            'agile': 'Agile', 'scrum': 'Scrum',
            'jira': 'Jira', 'confluence': 'Confluence',
            'git': 'Git',
            'data structures': 'Data Structures',
            'algorithms': 'Algorithms',
            'competitive coding': 'Competitive Coding',
            'dsa': 'Data Structures',
            'oop': 'OOP', 'object oriented programming': 'OOP',
            'android': 'Android', 'ios': 'iOS',
            'kotlin': 'Kotlin', 'swift': 'Swift',
            'react native': 'React Native', 'flutter': 'Flutter',
            'dart': 'Dart',
            'firebase': 'Firebase',
            'mobile app development': 'Mobile App Development',
            'full stack': 'Full Stack', 'full-stack': 'Full Stack',
            'frontend': 'Frontend', 'backend': 'Backend',
            'mern': 'MERN', 'mean': 'MEAN',
            'rest': 'REST API', 'graphql': 'GraphQL',
            'microservices': 'Microservices',
            'system design': 'System Design',
            'hld': 'System Design', 'lld': 'System Design',
            'cad design': 'CAD Design',
            'process simulation': 'Process Simulation',
            'manufacturing': 'Manufacturing',
            'structural engineering': 'Structural Engineering',
            'civil engineering': 'Civil Engineering',
            'mechanical engineering': 'Mechanical Engineering',
            'chemical engineering': 'Chemical Engineering',
            'electronics': 'Electronics',
            'telecommunications': 'Telecommunications',
            '5g': '5G', 'lte': 'LTE',
            'vlsi': 'VLSI', 'pcb design': 'PCB Design',
            'control systems': 'Control Systems',
            'power systems': 'Power Systems',
            'renewable energy': 'Renewable Energy',
            'ev technology': 'EV Technology',
            'automotive': 'Automotive',
            'aerospace': 'Aerospace',
            'aviation': 'Aviation',
            'materials science': 'Materials Science',
            'metallurgy': 'Metallurgy',
            'pharmaceuticals': 'Pharmaceuticals',
            'healthcare': 'Healthcare',
            'health tech': 'Health Tech',
            'medical devices': 'Medical Devices',
            'public health': 'Public Health',
            'mental health': 'Mental Health',
            'edtech': 'EdTech',
            'education': 'Education',
            'e-learning': 'E-Learning',
            'edutech': 'EdTech',
            'retail': 'Retail',
            'ecommerce': 'E-Commerce', 'e-commerce': 'E-Commerce',
            'supply chain': 'Supply Chain',
            'logistics': 'Logistics',
            'operations': 'Operations',
            'business development': 'Business Development',
            'sales': 'Sales',
            'marketing': 'Marketing',
            'digital marketing': 'Digital Marketing',
            'content writing': 'Content Writing',
            'technical writing': 'Technical Writing',
            'documentation': 'Documentation',
            'seo': 'SEO', 'sem': 'SEM',
            'social media': 'Social Media',
            'product management': 'Product Management',
            'product design': 'Product Design',
            'user research': 'User Research',
            'usability testing': 'Usability Testing',
            'user interface': 'User Interface',
            'user experience': 'User Experience',
            'game development': 'Game Development',
            'unity': 'Unity', 'unreal engine': 'Unreal Engine',
            'cocos2d': 'Cocos2d',
            'gamedev': 'Game Development',
            'ar': 'AR', 'vr': 'VR',
            'augmented reality': 'AR', 'virtual reality': 'VR',
            '3d modeling': '3D Modeling',
            'animation': 'Animation',
            'video editing': 'Video Editing',
            'graphic design': 'Graphic Design',
            'web design': 'Web Design',
            'motion graphics': 'Motion Graphics',
            'sound design': 'Sound Design',
            'music production': 'Music Production',
            'podcasting': 'Podcasting',
            'journalism': 'Journalism',
            'media': 'Media',
            'entertainment': 'Entertainment',
            'fashion': 'Fashion',
            'lifestyle': 'Lifestyle',
            'sports': 'Sports',
            'fitness': 'Fitness',
            'nutrition': 'Nutrition',
            'wellness': 'Wellness',
            'travel': 'Travel',
            'hospitality': 'Hospitality',
            'tourism': 'Tourism',
            'real estate': 'Real Estate',
            'property': 'Real Estate',
            'construction': 'Construction',
            'infrastructure': 'Infrastructure',
            'urban planning': 'Urban Planning',
            'architecture': 'Architecture',
            'interior design': 'Interior Design',
            'landscape': 'Landscape',
            'environmental': 'Environmental',
            'sustainability': 'Sustainability',
            'climate': 'Climate',
            'renewable': 'Renewable Energy',
            'clean energy': 'Clean Energy',
            'green tech': 'Green Tech',
            'water': 'Water Technology',
            'waste': 'Waste Tech',
            'air quality': 'Air Quality',
            'pollution': 'Environmental',
            'conservation': 'Environmental',
            'biodiversity': 'Environmental',
            'ecology': 'Environmental',
            'zoology': 'Environmental',
            'botany': 'Environmental',
            'biology': 'Biology',
            'chemistry': 'Chemistry',
            'physics': 'Physics',
            'mathematics': 'Mathematics',
            'statistics': 'Statistics',
            'econometrics': 'Statistics',
            'quantitative': 'Statistics',
            'qualitative': 'Research',
            'research': 'Research',
            'r&d': 'R&D',
            'innovation': 'R&D',
            'patents': 'R&D',
            'ip': 'R&D',
            'intellectual property': 'R&D',
            'legal': 'Legal',
            'law': 'Legal',
            'compliance': 'Compliance',
            'regulatory': 'Compliance',
            'risk': 'Risk Management',
            'audit': 'Audit',
            'accounting': 'Accounting',
            'tax': 'Tax',
            'treasury': 'Finance',
            'investment': 'Finance',
            'banking': 'Finance',
            'insurance': 'Finance',
            'wealth management': 'Finance',
            'asset management': 'Finance',
            'portfolio': 'Finance',
            'trading': 'Finance',
            'stocks': 'Finance',
            'bonds': 'Finance',
            'derivatives': 'Finance',
            'forex': 'Finance',
            'crypto': 'Blockchain',
            'defi': 'DeFi',
            'nft': 'NFT',
            'dao': 'Blockchain',
            'metaverse': 'Blockchain',
            'gamefi': 'Blockchain',
            'tokenomics': 'Blockchain'
        }

        self.tech_acronyms = {
            'nlp', 'aws', 'sql', 'css', 'html', 'js', 'api', 'ml', 'ai', 'iot', 'cv',
            'dl', 'rl', 'gan', 'rnn', 'cnn', 'svm', 'knn', 'pca', 'eda', 'gpu', 'cpu',
            'ram', 'os', 'ui', 'ux', 'devops', 'ci/cd', 'mlops', 'gis', 'gps', 'rfid',
            'nfc', '5g', 'lte', 'wifi', 'bluetooth', 'zigbee', 'mqtt', 'http', 'https',
            'tcp', 'udp', 'ip', 'dns', 'ssl', 'tls', 'oauth', 'jwt', 'json', 'xml',
            'yaml', 'csv', 'pdf', 'api', 'sdk', 'ide', 'cli', 'gui', 'ux', 'ui'
        }

        self.location_map = {
            'mumbai': 'Mumbai', 'bombay': 'Mumbai',
            'delhi': 'Delhi', 'new delhi': 'Delhi', 'delhi ncr': 'Delhi',
            'bangalore': 'Bangalore', 'bengaluru': 'Bangalore',
            'hyderabad': 'Hyderabad',
            'chennai': 'Chennai',
            'pune': 'Pune',
            'kolkata': 'Kolkata', 'calcutta': 'Kolkata',
            'ahmedabad': 'Ahmedabad',
            'jaipur': 'Jaipur',
            'noida': 'Noida', 'gurugram': 'Gurgaon', 'gurgaon': 'Gurgaon',
            'remote': 'Remote', 'work from home': 'Remote',
            'rural india': 'Rural India', 'village': 'Rural India'
        }

        self.domain_skill_weights = {
            'Data Science & Analytics': {
                'Python': 0.96, 'SQL': 0.92, 'Machine Learning': 0.84, 'Statistics': 0.83,
                'Pandas': 0.82, 'NumPy': 0.81, 'Scikit-learn': 0.78, 'TensorFlow': 0.62,
                'PyTorch': 0.68, 'Data Visualization': 0.72, 'Tableau': 0.58, 'Power BI': 0.56,
                'Data Analysis': 0.88, 'Statistical Analysis': 0.82, 'Deep Learning': 0.64,
                'NLP': 0.57, 'Apache Spark': 0.66, 'Hadoop': 0.48, 'Kafka': 0.45,
                'Excel': 0.52, 'MongoDB': 0.34, 'MySQL': 0.58, 'PostgreSQL': 0.64
            },
            'Artificial Intelligence': {
                'Machine Learning': 0.96, 'Deep Learning': 0.92, 'Python': 0.95,
                'TensorFlow': 0.72, 'PyTorch': 0.84, 'NLP': 0.86, 'Computer Vision': 0.84,
                'Natural Language Processing': 0.86, 'Data Analysis': 0.60,
                'Statistical Analysis': 0.68, 'SQL': 0.50, 'Java': 0.36, 'C++': 0.52
            },
            'Web Development': {
                'JavaScript': 0.96, 'React': 0.91, 'HTML': 0.90, 'CSS': 0.89,
                'Node.js': 0.84, 'Express.js': 0.76, 'MongoDB': 0.52, 'REST API': 0.82,
                'SQL': 0.72, 'Java': 0.44, 'Python': 0.48,
                'Docker': 0.58, 'AWS': 0.57, 'FastAPI': 0.46, 'Django': 0.54, 'Flask': 0.43
            },
            'Mobile Development': {
                'Java': 0.58, 'Kotlin': 0.86, 'Swift': 0.84,
                'Android': 0.88, 'iOS': 0.86, 'React Native': 0.70, 'Flutter': 0.78,
                'JavaScript': 0.54, 'Python': 0.24, 'Firebase': 0.66
            },
            'Cloud Computing': {
                'AWS': 0.95, 'Docker': 0.90, 'Kubernetes': 0.92, 'Azure': 0.82,
                'Google Cloud Platform': 0.76, 'Terraform': 0.82, 'CI/CD': 0.84,
                'Python': 0.60, 'Linux': 0.80, 'Bash': 0.72, 'Jenkins': 0.62,
                'GitLab': 0.56, 'GitHub': 0.62
            },
            'Finance': {
                'Excel': 0.95, 'Financial Modeling': 0.90, 'Data Analysis': 0.82,
                'SQL': 0.74, 'Python': 0.70, 'Tableau': 0.58, 'Power BI': 0.60,
                'Statistics': 0.66, 'Machine Learning': 0.42, 'Bloomberg Terminal': 0.72
            },
            'Software Engineering': {
                'Python': 0.82, 'Java': 0.80, 'SQL': 0.70, 'Data Structures': 0.92,
                'Algorithms': 0.92, 'Git': 0.84, 'Docker': 0.66, 'C++': 0.72,
                'JavaScript': 0.68
            },
            'Cybersecurity': {
                'Python': 0.62, 'Linux': 0.84, 'Network Security': 0.92,
                'Ethical Hacking': 0.80, 'Penetration Testing': 0.86, 'SIEM': 0.82,
                'Cybersecurity': 0.90, 'Cyber Security': 0.90, 'Security Analysis': 0.84
            },
            'IoT': {
                'Python': 0.56, 'C++': 0.82, 'Arduino': 0.76, 'Raspberry Pi': 0.74,
                'MQTT': 0.80, 'Sensors': 0.78, 'Embedded Systems': 0.90, 'Linux': 0.66,
                'Robotics': 0.60, 'ROS': 0.58, 'Circuit Design': 0.70
            },
            'Blockchain': {
                'Solidity': 0.94, 'Ethereum': 0.84, 'Web3.js': 0.76,
                'Smart Contracts': 0.92, 'Python': 0.34, 'JavaScript': 0.64,
                'Blockchain': 0.74, 'Cryptocurrency': 0.40
            },
            'Product Management': {
                'SQL': 0.66, 'Data Analysis': 0.76, 'Project Management': 0.92,
                'Figma': 0.52, 'Excel': 0.72, 'Business Analysis': 0.88
            },
            'DevOps': {
                'Docker': 0.92, 'Kubernetes': 0.94, 'Jenkins': 0.68, 'AWS': 0.86,
                'Terraform': 0.84, 'Python': 0.62, 'Linux': 0.82,
                'CI/CD': 0.94, 'GitLab': 0.64, 'GitHub': 0.70, 'Bash': 0.76
            },
            'Data Engineering': {
                'Python': 0.90, 'SQL': 0.96, 'Apache Spark': 0.88, 'Kafka': 0.82,
                'Hadoop': 0.54, 'Airflow': 0.82, 'Docker': 0.64, 'AWS': 0.78,
                'Data Engineering': 0.92
            },
            'Fintech': {
                'Python': 0.84, 'SQL': 0.78, 'Data Analysis': 0.80,
                'Machine Learning': 0.58, 'Blockchain': 0.42, 'Excel': 0.62,
                'Fintech': 0.86
            },
            'UI/UX Design': {
                'Figma': 0.96, 'UI Design': 0.90, 'UX Design': 0.92,
                'Prototyping': 0.86, 'Wireframing': 0.82, 'User Research': 0.84,
                'Adobe XD': 0.30, 'Sketch': 0.42, 'InVision': 0.22,
                'Photoshop': 0.48, 'Illustrator': 0.44, 'Design Thinking': 0.72
            },
            'Mechanical Engineering': {
                'AutoCAD': 0.84, 'SolidWorks': 0.90, 'ANSYS': 0.86,
                'CAD Design': 0.88, 'MATLAB': 0.62, 'Simulink': 0.52,
                'Robotics': 0.46, 'Manufacturing': 0.78
            },
            'Civil Engineering': {
                'AutoCAD': 0.88, 'STAAD.Pro': 0.84, 'Civil Engineering': 0.98,
                'Structural Engineering': 0.92, 'Construction Tech': 0.82,
                'Project Management': 0.64, 'Excel': 0.48
            },
            'Chemical Engineering': {
                'Aspen Plus': 0.92, 'Process Simulation': 0.90,
                'Chemical Engineering': 0.98, 'MATLAB': 0.58,
                'Data Analysis': 0.42, 'Excel': 0.38
            },
            'Biotechnology': {
                'Gel Electrophoresis': 0.78, 'Bioinformatics': 0.90,
                'Biotechnology': 0.98, 'PCR': 0.84, 'Data Analysis': 0.56,
                'Python': 0.48, 'Statistical Analysis': 0.68
            }
        }

        self.skill_clusters = {
            'ai_ml': {
                'Machine Learning', 'ML', 'Deep Learning', 'DL', 'Neural Networks',
                'Artificial Intelligence', 'AI', 'Reinforcement Learning', 'RL',
                'Computer Vision', 'CV', 'NLP', 'Natural Language Processing',
                'TensorFlow', 'PyTorch', 'Scikit-learn', 'Keras', 'XGBoost',
                'LightGBM', 'Transformers', 'BERT', 'GPT', 'Generative AI',
                'GANs', 'Diffusion Models', 'Time Series Analysis',
                'Statistical Analysis', 'Data Analysis', 'Predictive Modeling'
            },
            'web_dev': {
                'JavaScript', 'JS', 'TypeScript', 'TS', 'HTML', 'CSS',
                'React', 'ReactJS', 'React.js', 'Angular', 'Vue.js', 'Vue',
                'Node.js', 'NodeJS', 'Express.js', 'Express', 'Next.js',
                'Django', 'Flask', 'FastAPI', 'Spring Boot', 'ASP.NET',
                'MongoDB', 'MySQL', 'PostgreSQL', 'SQL', 'Redis',
                'REST API', 'GraphQL', 'Web Development', 'Full Stack',
                'Frontend', 'Backend', 'MERN', 'MEAN'
            },
            'data_science': {
                'Python', 'Py', 'R', 'SQL', 'Pandas', 'NumPy', 'Scikit-learn',
                'Data Analysis', 'Data Analytics', 'Data Visualization',
                'Tableau', 'Power BI', 'PowerBI', 'Matplotlib', 'Seaborn',
                'Statistical Analysis', 'Statistics', 'Excel',
                'Big Data', 'Hadoop', 'Apache Spark', 'Spark', 'Kafka',
                'Data Mining', 'Data Engineering', 'ETL', 'Data Warehousing'
            },
            'cloud_devops': {
                'AWS', 'Amazon Web Services', 'Azure', 'Microsoft Azure',
                'GCP', 'Google Cloud Platform', 'Google Cloud',
                'Docker', 'Kubernetes', 'K8s', 'Jenkins', 'GitLab', 'GitHub',
                'Terraform', 'Ansible', 'CI/CD', 'DevOps',
                'Linux', 'Bash', 'Shell Scripting', 'Cloud Computing'
            },
            'mobile_dev': {
                'Android', 'iOS', 'Kotlin', 'Swift', 'Java', 'Dart',
                'React Native', 'Flutter', 'Mobile Development',
                'Firebase', 'Mobile App', 'App Development'
            },
            'programming': {
                'Python', 'Py', 'Java', 'JavaScript', 'JS', 'TypeScript', 'TS',
                'C++', 'C#', 'C Sharp', 'Go', 'Golang', 'Rust', 'Ruby',
                'PHP', 'Swift', 'Kotlin', 'Dart', 'Scala', 'R', 'MATLAB',
                'Programming', 'Coding', 'Software Development',
                'Data Structures', 'Algorithms', 'Git', 'OOP'
            },
            'database': {
                'SQL', 'MySQL', 'PostgreSQL', 'Postgres', 'MongoDB', 'Mongo',
                'Redis', 'Cassandra', 'Oracle', 'SQLite', 'MariaDB',
                'NoSQL', 'Database', 'Database Management', 'DBMS'
            },
            'cybersecurity': {
                'Cybersecurity', 'Cyber Security', 'Network Security',
                'Ethical Hacking', 'Penetration Testing', 'Security',
                'Cryptography', 'SIEM', 'Firewall', 'Intrusion Detection',
                'Vulnerability Assessment', 'Security Analysis'
            },
            'iot_embedded': {
                'IoT', 'Internet of Things', 'Arduino', 'Raspberry Pi',
                'Sensors', 'MQTT', 'Zigbee', 'Bluetooth', 'WiFi',
                'Robotics', 'ROS', 'Automation', 'Hardware',
                'Embedded Systems', 'Circuit Design', 'VHDL'
            },
            'blockchain': {
                'Blockchain', 'Ethereum', 'Solidity', 'Smart Contracts',
                'Web3.js', 'Web3', 'Cryptocurrency', 'DeFi', 'NFT',
                'Hyperledger', 'Distributed Ledger'
            },
            'design_ux': {
                'Figma', 'Adobe XD', 'Sketch', 'InVision', 'Photoshop',
                'Illustrator', 'UI Design', 'UX Design', 'User Research',
                'Usability Testing', 'Wireframing', 'Prototyping',
                'User Interface', 'User Experience', 'Design Thinking',
                'Graphic Design', 'Web Design', 'Motion Graphics'
            },
            'project_mgmt': {
                'Project Management', 'Agile', 'Scrum', 'Kanban',
                'Jira', 'Confluence', 'Trello', 'Asana',
                'Product Management', 'Business Analysis', 'Stakeholder Management'
            },
            'communication': {
                'Communication Skills', 'Communication', 'Presentation',
                'Public Speaking', 'Technical Writing', 'Documentation',
                'Leadership', 'Teamwork', 'Collaboration'
            },
            'finance_fintech': {
                'Excel', 'Financial Modeling', 'Finance', 'Fintech',
                'Bloomberg Terminal', 'Data Analysis', 'Statistics',
                'Investment', 'Banking', 'Trading', 'Portfolio',
                'Risk Management', 'Accounting', 'Tax', 'Audit'
            },
            'mechanical_engineering': {
                'AutoCAD', 'SolidWorks', 'ANSYS', 'CAD Design',
                'MATLAB', 'Simulink', 'Mechanical Engineering',
                'Manufacturing', 'Robotics', 'Automotive', 'Aerospace'
            },
            'civil_engineering': {
                'AutoCAD', 'STAAD.Pro', 'Civil Engineering',
                'Structural Engineering', 'Construction Tech',
                'Urban Planning', 'Architecture', 'Infrastructure'
            },
            'chemical_engineering': {
                'Aspen Plus', 'Process Simulation', 'Chemical Engineering',
                'MATLAB', 'Chemistry', 'Materials Science', 'Metallurgy'
            },
            'biotechnology': {
                'Gel Electrophoresis', 'Bioinformatics', 'Biotechnology',
                'PCR', 'Biology', 'Pharmaceuticals', 'Healthcare',
                'Medical Devices', 'Public Health', 'Mental Health'
            },
            'electronics': {
                'Electronics', 'VLSI', 'PCB Design', 'Control Systems',
                'Power Systems', 'Telecommunications', '5G', 'LTE',
                'Embedded Systems', 'Circuit Design', 'Sensors'
            },
            'game_dev': {
                'Game Development', 'Unity', 'Unreal Engine', 'Cocos2d',
                'AR', 'VR', 'Augmented Reality', 'Virtual Reality',
                '3D Modeling', 'Animation', 'Sound Design'
            },
            'digital_marketing': {
                'Digital Marketing', 'SEO', 'SEM', 'Social Media',
                'Content Writing', 'Marketing', 'Sales',
                'E-Commerce', 'Retail', 'Business Development'
            },
            'research': {
                'Research', 'R&D', 'Innovation', 'Patents',
                'Technical Writing', 'Documentation', 'Data Analysis'
            }
        }

        self.domain_similarity_groups = {
            'ai_ml_group': {
                'Artificial Intelligence', 'Data Science & Analytics', 'AI',
                'Machine Learning', 'Deep Learning', 'NLP',
                'Computer Vision', 'Data Science', 'Analytics',
                'Healthcare AI', 'AI Governance', 'Generative AI'
            },
            'web_mobile_group': {
                'Web Development', 'Mobile Development', 'Full Stack Development',
                'Frontend Development', 'Backend Development', 'Software Engineering',
                'App Development', 'Game Development', 'Software Development'
            },
            'data_group': {
                'Data Engineering', 'Business Intelligence', 'Data Science',
                'Retail Analytics', 'Sports Analytics', 'Media Analytics', 'Legal Analytics',
                'Analytics', 'Database Management'
            },
            'cloud_infra_group': {
                'Cloud Computing', 'DevOps', 'Site Reliability Engineering', 'CI/CD',
                'Infrastructure', 'Platform Engineering', 'SRE', 'Cloud Services'
            },
            'finance_group': {
                'Finance', 'Fintech', 'Quantitative Finance', 'Financial Services',
                'Banking', 'Investment Banking', 'Financial Analysis'
            },
            'iot_hardware_group': {
                'IoT', 'Robotics', 'Embedded Systems', 'Hardware Engineering',
                'Electronics', 'Automation', 'Smart Cities', 'EV Technology'
            },
            'security_group': {
                'Cybersecurity', 'Information Security', 'Network Security',
                'Security Analysis', 'Ethical Hacking', 'Security'
            },
            'blockchain_group': {
                'Blockchain', 'Cryptocurrency', 'DeFi', 'Web3',
                'Smart Contracts', 'Distributed Systems'
            },
            'design_group': {
                'UX Research', 'UI/UX Design', 'Product Design', 'Design',
                'Creative Technology', 'User Research'
            },
            'research_group': {
                'Research', 'R&D', 'Academic Research', 'Scientific Research',
                'EdTech Research', 'Policy Research'
            },
            'sustainability_group': {
                'Sustainability', 'Environmental Tech', 'Climate Tech',
                'Renewable Energy', 'Clean Energy', 'Green Tech',
                'Water Technology', 'Waste Tech', 'Air Quality'
            },
            'healthcare_group': {
                'Healthcare', 'Health Tech', 'Medical Devices',
                'Public Health', 'Mental Health', 'Maternal Health'
            },
            'education_group': {
                'EdTech', 'Education', 'Learning', 'Training',
                'Curriculum Design'
            },
            'agriculture_group': {
                'AgriTech', 'Agriculture', 'Farming', 'Food Tech',
                'Digital Agriculture', 'Precision Agriculture'
            },
            'supply_chain_group': {
                'Supply Chain', 'Logistics', 'Logistics Technology',
                'Operations', 'Transportation', 'Warehouse Management'
            },
            'real_estate_group': {
                'PropTech', 'Real Estate', 'Real Estate Tech',
                'Construction Tech', 'Urban Planning'
            },
            'media_group': {
                'Media', 'Entertainment', 'Music Technology',
                'Fashion Technology', 'Performing Arts',
                'Dance Technology', 'Cultural Heritage', 'Marketing',
                'Digital Marketing'
            },
            'policy_group': {
                'Public Policy', 'Policy Innovation',
                'Governance', 'Public Administration'
            },
            'accessibility_group': {
                'Accessibility', 'Assistive Technology', 'Disability Tech',
                'Inclusion Tech', 'Accessibility Tools'
            },
            'aerospace_group': {
                'Aerospace', 'Aviation', 'Space Systems'
            },
            'biology_group': {
                'Computational Biology', 'Biotechnology', 'Bioinformatics', 'Pharmaceuticals'
            },
            'energy_group': {
                'Energy Tech', 'Renewable Energy'
            },
            'transportation_group': {
                'Transportation Tech', 'Electric Mobility', 'Public Transport', 'Traffic Management'
            },
            'materials_group': {
                'Materials Science', 'Metallurgy', 'Chemical Engineering'
            },
            'civil_engineering_group': {
                'Civil Engineering', 'Structural Engineering', 'Geotechnical Engineering'
            },
            'mechanical_engineering_group': {
                'Mechanical Engineering', 'Manufacturing', 'Automotive', 'Aerospace',
                'Robotics', 'EV Technology'
            },
            'chemical_engineering_group': {
                'Chemical Engineering', 'Process Engineering', 'Materials Science',
                'Pharmaceuticals', 'Energy Tech'
            },
            'biotechnology_group': {
                'Biotechnology', 'Bioinformatics', 'Computational Biology',
                'Pharmaceuticals', 'Healthcare', 'Health Tech'
            },
            'product_management_group': {
                'Product Management', 'Business Analysis', 'Project Management',
                'Business Intelligence'
            },
            'qa_testing_group': {
                'Quality Assurance', 'Testing', 'Test Engineering', 'QA',
                'Software Testing', 'Test Automation'
            }
        }

        self.interest_to_domain = {
            'data science': 'Data Science & Analytics',
            'machine learning': 'Artificial Intelligence',
            'ai': 'Artificial Intelligence',
            'artificial intelligence': 'Artificial Intelligence',
            'web development': 'Web Development',
            'full stack': 'Web Development',
            'frontend': 'Web Development',
            'backend': 'Web Development',
            'mobile development': 'Mobile Development',
            'android': 'Mobile Development',
            'ios': 'Mobile Development',
            'cloud': 'Cloud Computing',
            'devops': 'Cloud Computing',
            'cybersecurity': 'Cybersecurity',
            'ethical hacking': 'Cybersecurity',
            'blockchain': 'Blockchain',
            'iot': 'IoT',
            'robotics': 'IoT',
            'electronics': 'Electronics & Communication',
            'mechanical': 'Mechanical Engineering',
            'civil': 'Civil Engineering',
            'chemical': 'Chemical Engineering',
            'biotechnology': 'Biotechnology',
            'finance': 'Finance',
            'economics': 'Finance',
            'ui/ux': 'UI/UX Design',
            'ui ux': 'UI/UX Design',
            'design': 'UI/UX Design',
            'figma': 'UI/UX Design',
            'product design': 'UI/UX Design',
            'game dev': 'Software Engineering',
            'game development': 'Software Engineering',
            'competitive coding': 'Software Engineering',
            'dsa': 'Software Engineering',
            'data structures': 'Software Engineering',
            'algorithms': 'Software Engineering'
        }

        # Branch-domain relevance favors curriculum overlap plus market adjacency:
        # 1.0 = direct placement pathway, 0.7-0.9 = strong adjacent pathway, 0.3-0.6 = transferable.
        self.branch_domain_relevance = {
            'Computer Science': {
                'Data Science & Analytics': 1.0, 'Artificial Intelligence': 1.0,
                'Web Development': 1.0, 'Mobile Development': 1.0, 'Cloud Computing': 1.0,
                'Cybersecurity': 1.0, 'Blockchain': 1.0, 'IoT': 1.0,
                'Software Engineering': 1.0, 'DevOps': 1.0, 'Data Engineering': 1.0,
                'Electronics & Communication': 0.35, 'Civil Engineering': 0.18,
                'Mechanical Engineering': 0.20, 'Chemical Engineering': 0.18,
                'Biotechnology': 0.28, 'Finance': 0.45, 'Architecture': 0.22,
                'Fintech': 0.62, 'Retail Analytics': 0.60, 'Healthcare AI': 0.68,
                'UI/UX Design': 0.65
            },
            'IT': {
                'Data Science & Analytics': 1.0, 'Artificial Intelligence': 1.0,
                'Web Development': 1.0, 'Mobile Development': 1.0, 'Cloud Computing': 1.0,
                'Cybersecurity': 1.0, 'Blockchain': 1.0, 'IoT': 1.0,
                'Software Engineering': 1.0, 'DevOps': 1.0, 'Data Engineering': 1.0,
                'Electronics & Communication': 0.34, 'Civil Engineering': 0.18,
                'Mechanical Engineering': 0.20, 'Chemical Engineering': 0.18,
                'Biotechnology': 0.28, 'Finance': 0.44, 'Architecture': 0.22,
                'UI/UX Design': 0.62
            },
            'Data Science': {
                'Data Science & Analytics': 1.0, 'Artificial Intelligence': 0.95,
                'Data Engineering': 0.96, 'Retail Analytics': 0.92, 'Sports Analytics': 0.88,
                'Media Analytics': 0.86, 'Legal Analytics': 0.82, 'Finance': 0.76,
                'Web Development': 0.36, 'Mobile Development': 0.22, 'Cloud Computing': 0.66,
                'Cybersecurity': 0.34, 'Blockchain': 0.42, 'IoT': 0.32,
                'Electronics & Communication': 0.20, 'Civil Engineering': 0.18,
                'Mechanical Engineering': 0.18, 'Chemical Engineering': 0.20,
                'Biotechnology': 0.50, 'Architecture': 0.18, 'UI/UX Design': 0.32
            },
            'AI': {
                'Data Science & Analytics': 0.95, 'Artificial Intelligence': 1.0,
                'Machine Learning': 1.0, 'Deep Learning': 1.0, 'NLP': 1.0,
                'Computer Vision': 0.98, 'Healthcare AI': 0.86, 'AI Governance': 0.72,
                'Generative AI': 1.0, 'Robotics': 0.76, 'IoT': 0.54,
                'Data Engineering': 0.74, 'Web Development': 0.34, 'Mobile Development': 0.24,
                'Cloud Computing': 0.64, 'Cybersecurity': 0.38, 'Blockchain': 0.28,
                'Electronics & Communication': 0.34, 'Civil Engineering': 0.16,
                'Mechanical Engineering': 0.18, 'Chemical Engineering': 0.18,
                'Biotechnology': 0.42, 'Finance': 0.34, 'Architecture': 0.16,
                'UI/UX Design': 0.28
            },
            'Electronics & Communication': {
                'IoT': 1.0, 'Electronics': 1.0, 'Robotics': 0.74, 'Embedded Systems': 1.0,
                'Hardware Engineering': 0.98, 'Telecommunications': 0.98,
                'Data Science & Analytics': 0.42, 'Artificial Intelligence': 0.54,
                'Web Development': 0.24, 'Mobile Development': 0.22, 'Cloud Computing': 0.38,
                'Cybersecurity': 0.56, 'Blockchain': 0.20,
                'Civil Engineering': 0.16, 'Mechanical Engineering': 0.42,
                'Chemical Engineering': 0.16, 'Biotechnology': 0.24, 'Finance': 0.18, 'Architecture': 0.18,
                'UI/UX Design': 0.24
            },
            'Electrical Engineering': {
                'IoT': 0.72, 'Robotics': 0.72, 'Embedded Systems': 0.68, 'EV Technology': 0.92,
                'Power Systems': 1.0, 'Energy Tech': 0.94, 'Electronics': 0.76,
                'Data Science & Analytics': 0.34, 'Artificial Intelligence': 0.42,
                'Web Development': 0.16, 'Mobile Development': 0.14, 'Cloud Computing': 0.26,
                'Civil Engineering': 0.28, 'Mechanical Engineering': 0.62,
                'Chemical Engineering': 0.28, 'Biotechnology': 0.18, 'Finance': 0.18, 'Architecture': 0.24,
                'UI/UX Design': 0.16
            },
            'Civil Engineering': {
                'Civil Engineering': 1.0, 'Structural Engineering': 1.0, 'Geotechnical Engineering': 0.96,
                'Construction Tech': 0.94, 'Urban Planning': 0.86, 'PropTech': 0.76,
                'Architecture': 0.78, 'Transportation Tech': 0.72, 'Smart Cities': 0.66,
                'Data Science & Analytics': 0.34, 'Artificial Intelligence': 0.24,
                'Web Development': 0.14, 'Mobile Development': 0.12, 'Cloud Computing': 0.18,
                'Mechanical Engineering': 0.42, 'Chemical Engineering': 0.24,
                'Biotechnology': 0.14, 'Finance': 0.26, 'Electronics & Communication': 0.22,
                'UI/UX Design': 0.28
            },
            'Mechanical Engineering': {
                'Mechanical Engineering': 1.0, 'Robotics': 0.74, 'EV Technology': 0.76,
                'Aerospace': 0.84, 'Automotive': 0.96, 'Manufacturing': 0.98,
                'Civil Engineering': 0.46, 'Electrical Engineering': 0.56,
                'Data Science & Analytics': 0.30, 'Artificial Intelligence': 0.34,
                'Web Development': 0.12, 'Mobile Development': 0.10, 'Cloud Computing': 0.16,
                'Chemical Engineering': 0.38, 'Biotechnology': 0.18, 'Finance': 0.22, 'Architecture': 0.30,
                'UI/UX Design': 0.18
            },
            'Chemical Engineering': {
                'Chemical Engineering': 1.0, 'Biotechnology': 0.76, 'Pharmaceuticals': 0.84,
                'Materials Science': 0.82, 'Energy Tech': 0.68, 'Environmental Tech': 0.72,
                'Mechanical Engineering': 0.46, 'Civil Engineering': 0.34,
                'Data Science & Analytics': 0.30, 'Artificial Intelligence': 0.24,
                'Web Development': 0.10, 'Mobile Development': 0.08, 'Cloud Computing': 0.16,
                'Electrical Engineering': 0.30, 'Electronics & Communication': 0.22,
                'Finance': 0.20, 'Architecture': 0.16,
                'UI/UX Design': 0.12
            },
            'Biotechnology': {
                'Biotechnology': 1.0, 'Computational Biology': 0.95, 'Bioinformatics': 1.0,
                'Pharmaceuticals': 0.86, 'Healthcare': 0.84, 'Healthcare AI': 0.76,
                'Data Science & Analytics': 0.66, 'Artificial Intelligence': 0.50,
                'Chemical Engineering': 0.68, 'Mechanical Engineering': 0.16,
                'Web Development': 0.10, 'Mobile Development': 0.08, 'Cloud Computing': 0.20,
                'Electrical Engineering': 0.18, 'Electronics & Communication': 0.18,
                'Civil Engineering': 0.12, 'Finance': 0.14, 'Architecture': 0.10,
                'UI/UX Design': 0.12
            },
            'Architecture': {
                'Architecture': 1.0, 'Civil Engineering': 0.76, 'Urban Planning': 0.88,
                'PropTech': 0.72, 'Construction Tech': 0.78, 'Design': 1.0,
                'Data Science & Analytics': 0.18, 'Artificial Intelligence': 0.22,
                'Web Development': 0.22, 'Mobile Development': 0.16, 'Cloud Computing': 0.14,
                'Mechanical Engineering': 0.32, 'Chemical Engineering': 0.14,
                'Biotechnology': 0.08, 'Finance': 0.18, 'Electronics & Communication': 0.14,
                'UI/UX Design': 0.74
            },
            'Finance': {
                'Finance': 1.0, 'Fintech': 0.92, 'Quantitative Finance': 1.0, 'Banking': 0.98,
                'Data Science & Analytics': 0.72, 'Retail Analytics': 0.58,
                'Blockchain': 0.48, 'Artificial Intelligence': 0.42,
                'Web Development': 0.14, 'Mobile Development': 0.12, 'Cloud Computing': 0.18,
                'Cybersecurity': 0.36, 'IoT': 0.08,
                'Electronics & Communication': 0.12, 'Civil Engineering': 0.10,
                'Mechanical Engineering': 0.10, 'Chemical Engineering': 0.10,
                'Biotechnology': 0.12, 'Architecture': 0.10, 'UI/UX Design': 0.20
            },
            'Electronics': {
                'Electronics': 1.0, 'IoT': 0.88, 'Robotics': 0.72, 'Embedded Systems': 0.92,
                'Hardware Engineering': 1.0, 'Electronics & Communication': 0.92,
                'Data Science & Analytics': 0.38, 'Artificial Intelligence': 0.50,
                'Web Development': 0.20, 'Mobile Development': 0.18, 'Cloud Computing': 0.34,
                'Cybersecurity': 0.50, 'Blockchain': 0.18,
                'Civil Engineering': 0.14, 'Mechanical Engineering': 0.40,
                'Chemical Engineering': 0.14, 'Biotechnology': 0.20, 'Finance': 0.16, 'Architecture': 0.14,
                'UI/UX Design': 0.20
            }
        }

        self.cluster_compatibility = {
            'ai_ml': {'ai_ml', 'data_science', 'programming'},
            'data_science': {'data_science', 'ai_ml', 'programming', 'database'},
            'web_dev': {'web_dev', 'programming', 'database', 'cloud_devops'},
            'cloud_devops': {'cloud_devops', 'programming', 'web_dev', 'iot_embedded'},
            'mobile_dev': {'mobile_dev', 'web_dev', 'programming'},
            'programming': {'programming', 'web_dev', 'data_science', 'ai_ml', 'cloud_devops'},
            'database': {'database', 'data_science', 'web_dev', 'programming'},
            'cybersecurity': {'cybersecurity', 'cloud_devops', 'iot_embedded', 'programming'},
            'iot_embedded': {'iot_embedded', 'robotics', 'cloud_devops', 'programming', 'electronics'},
            'blockchain': {'blockchain', 'programming', 'database'},
            'design_ux': {'design_ux', 'web_dev', 'mobile_dev'},
            'project_mgmt': {'project_mgmt', 'web_dev', 'data_science', 'cloud_devops'},
            'communication': {'communication', 'project_mgmt', 'design_ux'},
            'finance_fintech': {'finance_fintech', 'data_science', 'project_mgmt', 'blockchain'},
            'mechanical_engineering': {'mechanical_engineering', 'iot_embedded', 'electronics'},
            'civil_engineering': {'civil_engineering', 'design_ux', 'project_mgmt'},
            'chemical_engineering': {'chemical_engineering', 'biotechnology', 'data_science'},
            'biotechnology': {'biotechnology', 'chemical_engineering', 'data_science'},
            'electronics': {'electronics', 'iot_embedded', 'programming'},
            'game_dev': {'game_dev', 'programming', 'design_ux'},
            'digital_marketing': {'digital_marketing', 'communication', 'project_mgmt'},
            'research': {'research', 'data_science', 'biotechnology', 'chemical_engineering'}
        }

        # Fallback weights approximate domain fit when an exact skill is missing but a related
        # cluster is present. These are intentionally more conservative than direct skill weights.
        self.domain_fallback_weights = { ## It helps the system give partial score for related skills when exact skills are missing.
            'Data Science & Analytics': {'data_science': 0.68, 'ai_ml': 0.58, 'programming': 0.48, 'database': 0.56, 'default': 0.18},
            'Artificial Intelligence': {'ai_ml': 0.72, 'data_science': 0.58, 'programming': 0.44, 'default': 0.18},
            'Web Development': {'web_dev': 0.72, 'programming': 0.58, 'database': 0.50, 'design_ux': 0.36, 'default': 0.16},
            'Mobile Development': {'mobile_dev': 0.72, 'web_dev': 0.54, 'programming': 0.46, 'default': 0.16},
            'Cloud Computing': {'cloud_devops': 0.74, 'programming': 0.46, 'iot_embedded': 0.26, 'default': 0.16},
            'Cybersecurity': {'cybersecurity': 0.78, 'cloud_devops': 0.48, 'iot_embedded': 0.24, 'programming': 0.34, 'default': 0.10},
            'IoT': {'iot_embedded': 0.76, 'cloud_devops': 0.34, 'programming': 0.30, 'electronics': 0.62, 'default': 0.14},
            'Blockchain': {'blockchain': 0.76, 'programming': 0.46, 'database': 0.36, 'default': 0.14},
            'Finance': {'finance_fintech': 0.72, 'project_mgmt': 0.40, 'data_science': 0.48, 'default': 0.16},
            'Software Engineering': {'programming': 0.78, 'web_dev': 0.56, 'cloud_devops': 0.42, 'default': 0.22},
            'Data Engineering': {'data_science': 0.68, 'database': 0.74, 'programming': 0.52, 'cloud_devops': 0.44, 'default': 0.16},
            'Fintech': {'finance_fintech': 0.76, 'data_science': 0.52, 'blockchain': 0.32, 'programming': 0.34, 'default': 0.16},
            'DevOps': {'cloud_devops': 0.82, 'programming': 0.38, 'default': 0.14},
            'Product Management': {'project_mgmt': 0.76, 'data_science': 0.46, 'design_ux': 0.34, 'default': 0.16},
            'UI/UX Design': {'design_ux': 0.82, 'web_dev': 0.38, 'mobile_dev': 0.28, 'default': 0.14},
            'Mechanical Engineering': {'mechanical_engineering': 0.78, 'iot_embedded': 0.32, 'electronics': 0.36, 'default': 0.14},
            'Civil Engineering': {'civil_engineering': 0.80, 'design_ux': 0.18, 'project_mgmt': 0.34, 'default': 0.14},
            'Chemical Engineering': {'chemical_engineering': 0.80, 'biotechnology': 0.44, 'research': 0.38, 'default': 0.14},
            'Biotechnology': {'biotechnology': 0.82, 'chemical_engineering': 0.42, 'research': 0.46, 'default': 0.14}
        }

        self.cleaning_stats = {
            'students_loaded': 0,
            'students_removed': 0,
            'students_kept': 0,
            'internships_loaded': 0,
            'internships_removed': 0,
            'internships_kept': 0,
            'removal_reasons': defaultdict(int)
        }

        self.removed_students = []
        self.removed_internships = []
        self.students_removal_reasons = defaultdict(list)
        self.internships_removal_reasons = defaultdict(list)

        self._skill_cache = {}
        self._branch_cache = {}
        self._location_cache = {}

    def _is_missing(self, val: Any) -> bool:
        if pd.isna(val) or val is None or val == '':
            return True
        str_val = str(val).strip().lower()
        return str_val in self.missing_vals or str_val == 'nan' or str_val == 'none'

    def _record_removal_reason(self, reason: str, dataset_type: str = None, row_id: str = None) -> None:
        self.cleaning_stats['removal_reasons'][reason] += 1

        if dataset_type == 'students' and row_id:
            self.removed_students.append((row_id, reason))
            self.students_removal_reasons[reason].append(row_id)
        elif dataset_type == 'internships' and row_id:
            self.removed_internships.append((row_id, reason))
            self.internships_removal_reasons[reason].append(row_id)

    def _clean_text(self, val: Any) -> str:
        if self._is_missing(val):
            return ''
        text = str(val).strip()
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\xa0', ' ')
        return text

    def _validate_number(self, val: Any, min_v: float, max_v: float, is_stipend: bool = False) -> Optional[float]:
        if self._is_missing(val):
            return None

        str_val = str(val).strip()
        if str_val.endswith('%'):
            try:
                num = float(str_val[:-1].strip())

                if is_stipend:
                    if num < 5000:
                        return None
                    return num if min_v <= num <= max_v else None

                if num > 100:
                    num = num / 100
                return num if min_v <= num <= max_v else None
            except (ValueError, TypeError):
                return None

        try:
            num = float(str_val)

            if is_stipend and num < 5000:
                return None

            return num if min_v <= num <= max_v else None
        except (ValueError, TypeError):
            return None

    def _normalize_branch(self, branch: Any) -> Optional[str]:
        if self._is_missing(branch):
            return None

        key = str(branch).strip().lower()
        key = re.sub(r'[^a-z0-9\s]', ' ', key).strip()
        key = re.sub(r'\s+', ' ', key)

        if key in self._branch_cache:
            return self._branch_cache[key]

        if key in self.branch_map:
            normalized = self.branch_map[key]
            self._branch_cache[key] = normalized
            return normalized

        matches = process.extract(key, self.branch_map.keys(), limit=1, scorer=fuzz.token_sort_ratio)
        if matches and matches[0][1] >= 80:
            normalized = self.branch_map[matches[0][0]]
            self._branch_cache[key] = normalized
            return normalized

        normalized = ' '.join(word.capitalize() for word in key.split())
        self._branch_cache[key] = normalized
        return normalized

    def _normalize_tier(self, tier: Any) -> Optional[str]:
        if self._is_missing(tier):
            return None

        key = str(tier).strip().lower()
        key = re.sub(r'[^a-z0-9-]', '', key)

        return self.tier_map.get(key)

    def _normalize_work_type(self, work_type: Any) -> Optional[str]:
        if self._is_missing(work_type):
            return None

        key = str(work_type).strip().lower()
        key = re.sub(r'[^a-z]', '', key)

        if not key:
            return None

        if key in self.work_type_map:
            return self.work_type_map[key]

        matches = process.extract(key, self.work_type_map.keys(), limit=1, scorer=fuzz.ratio)
        if matches and matches[0][1] >= 75:
            return self.work_type_map[matches[0][0]]

        if any(word in key for word in ['remote', 'wfh', 'home']):
            return 'Remote'
        elif 'hybrid' in key:
            return 'Hybrid'
        else:
            return None

    def _standardize_skill(self, skill: str) -> str:
        if not skill or self._is_missing(skill):
            return ''

        original_skill = skill.strip()
        cleaned_skill = re.sub(r'[^a-zA-Z0-9\s+&/-]', ' ', original_skill.lower())
        cleaned_skill = re.sub(r'\s+', ' ', cleaned_skill).strip()

        if not cleaned_skill:
            return ''

        cache_key = cleaned_skill.lower()
        if cache_key in self._skill_cache:
            return self._skill_cache[cache_key]

        if cleaned_skill in self.skill_standardization:
            standardized = self.skill_standardization[cleaned_skill]
            self._skill_cache[cache_key] = standardized
            return standardized

        if cleaned_skill in self.tech_acronyms:
            standardized = cleaned_skill.upper()
            self._skill_cache[cache_key] = standardized
            return standardized

        words = cleaned_skill.split()
        original_words = re.findall(r'[a-zA-Z0-9+&/-]+', original_skill)

        if len(words) > 1:
            if '&' in cleaned_skill or '+' in cleaned_skill:
                standardized = ' '.join(
                    word.upper() if word.lower() in self.tech_acronyms else word.capitalize()
                    for word in words
                )
            else:
                standardized = ' '.join(word.upper() if word.lower() in self.tech_acronyms else word.capitalize() for word in words)
        else:
            original_word = original_words[0] if original_words else cleaned_skill
            if cleaned_skill in self.tech_acronyms:
                standardized = cleaned_skill.upper()
            elif original_word.lower() in self.tech_acronyms:
                standardized = original_word.upper()
            else:
                standardized = cleaned_skill.capitalize()

        self._skill_cache[cache_key] = standardized
        return standardized

    def _get_skill_cluster_match_score(self, student_skills: set, internship_skills: set) -> float:
        if not internship_skills:
            return 0.0

        student_skills_lower = {s.lower() for s in student_skills}
        internship_skills_lower = {s.lower() for s in internship_skills}

        skill_to_cluster = {}
        for cluster_name, skills in self.skill_clusters.items():
            for skill in skills:
                skill_to_cluster[skill.lower()] = cluster_name

        exact_matches = student_skills_lower.intersection(internship_skills_lower)
        exact_match_count = len(exact_matches)

        matched_internship_skills = set(exact_matches)

        for intern_skill in internship_skills_lower:
            if intern_skill in exact_matches:
                continue

            intern_cluster = skill_to_cluster.get(intern_skill)
            if not intern_cluster:
                continue

            for student_skill in student_skills_lower:
                student_cluster = skill_to_cluster.get(student_skill)
                if student_cluster == intern_cluster:
                    matched_internship_skills.add(intern_skill)
                    break

        total_internship_skills = len(internship_skills)
        semantic_match_count = len(matched_internship_skills)

        score = semantic_match_count / total_internship_skills

        return min(1.0, score)

    def _get_domain_similarity_score(self, student_interests: str, internship_domain: str) -> float:
        if not student_interests:
            return 0.0

        interest_tokens = set()
        for token in re.split(r'[,\s]+', student_interests.lower()):
            token = token.strip()
            if len(token) > 1:
                interest_tokens.add(token)

        intern_domain_lower = internship_domain.lower()

        domain_to_group = {}
        for group_name, domains in self.domain_similarity_groups.items():
            for domain in domains:
                domain_to_group[domain.lower()] = group_name

        intern_group = domain_to_group.get(intern_domain_lower)

        matched_groups = set()
        exact_domain_match = False

        for interest_keyword, mapped_domain in self.interest_to_domain.items():
            if interest_keyword in interest_tokens:
                mapped_domain_lower = mapped_domain.lower()

                if mapped_domain_lower == intern_domain_lower:
                    exact_domain_match = True
                    break

                mapped_group = domain_to_group.get(mapped_domain_lower)

                if mapped_group:
                    matched_groups.add(mapped_group)

        if exact_domain_match:
            return 1.0

        if intern_group and intern_group in matched_groups:
            return 1.0

        if intern_domain_lower in interest_tokens:
            return 1.0

        if intern_group:
            for group_name, domains in self.domain_similarity_groups.items():
                if group_name == intern_group:
                    for domain in domains:
                        if domain.lower() in interest_tokens:
                            return 0.9

        return 0.0

    def _get_location_match_level(self, student_location: str, internship_location: str) -> float:
        if internship_location.lower() == 'remote':
            return 1.0

        if student_location.lower() == 'remote':
            return 1.0

        if student_location.lower() == internship_location.lower():
            return 1.0

        if student_location == 'Other City' or internship_location == 'Other City':
            return 0.4

        if student_location == 'Rural India' or internship_location == 'Rural India':
            return 0.4

        city_groups = {
            'ncr': {'delhi', 'noida', 'gurgaon', 'gurugram', 'faridabad', 'ghaziabad'},
            'maharashtra': {'mumbai', 'pune', 'nagpur', 'thane', 'navi mumbai'},
            'karnataka': {'bangalore', 'bengaluru', 'mysore', 'hubli'},
            'tamil_nadu': {'chennai', 'coimbatore', 'madurai'},
            'telangana': {'hyderabad', 'secunderabad', 'warangal'},
            'west_bengal': {'kolkata', 'howrah', 'durgapur'},
            'gujarat': {'ahmedabad', 'surat', 'vadodara', 'rajkot'},
            'rajasthan': {'jaipur', 'jodhpur', 'udaipur', 'kota'}
        }

        student_city = student_location.lower()
        intern_city = internship_location.lower()

        for group_name, cities in city_groups.items():
            if student_city in cities and intern_city in cities:
                return 0.7

        return 0.4

    def _infer_interests_from_skills(self, skills: List[str]) -> str:
        if not skills:
            return ''

        skill_to_interest = {
            'machine learning': 'machine learning',
            'deep learning': 'deep learning',
            'tensorflow': 'artificial intelligence',
            'pytorch': 'artificial intelligence',
            'keras': 'artificial intelligence',
            'python': 'data science',
            'sql': 'data science',
            'pandas': 'data science',
            'numpy': 'data science',
            'react': 'web development',
            'javascript': 'web development',
            'node.js': 'web development',
            'django': 'web development',
            'flask': 'web development',
            'html': 'web development',
            'css': 'web development',
            'aws': 'cloud computing',
            'docker': 'devops',
            'kubernetes': 'devops',
            'azure': 'cloud computing',
            'finance': 'finance',
            'financial modeling': 'finance',
            'excel': 'finance',
        }

        interest_counts = defaultdict(int)
        for skill in skills:
            skill_lower = skill.lower()
            for skill_keyword, interest in skill_to_interest.items():
                if skill_keyword in skill_lower:
                    interest_counts[interest] += 1

        if interest_counts:
            return max(interest_counts.keys(), key=lambda k: interest_counts[k])

        return ''

    def _get_domain_for_interest(self, interest: str) -> Optional[str]:
        if not interest:
            return None

        interest_lower = interest.lower()
        for interest_keyword, domain in self.interest_to_domain.items():
            if interest_keyword in interest_lower:
                return domain

        return None

    def _get_primary_skill_cluster(self, skills: List[str]) -> Optional[str]:
        if not skills:
            return None

        cluster_counts = defaultdict(int)

        skill_to_cluster = {}
        for cluster_name, cluster_skills in self.skill_clusters.items():
            for skill in cluster_skills:
                skill_to_cluster[skill.lower()] = cluster_name

        for skill in skills:
            skill_lower = skill.lower()
            cluster = skill_to_cluster.get(skill_lower)
            if cluster:
                cluster_counts[cluster] += 1

        if cluster_counts:
            return max(cluster_counts.keys(), key=lambda k: cluster_counts[k])

        return None

    def _are_clusters_compatible(self, cluster1: Optional[str], cluster2: Optional[str]) -> bool:
        if not cluster1 or not cluster2:
            return False

        compatible_clusters = self.cluster_compatibility.get(cluster1, {cluster1})
        return cluster2 in compatible_clusters

    def _get_skill_weight_with_fallback(self, skill: str, domain: str, skill_cluster: Optional[str]) -> float:
        domain_weights = self.domain_skill_weights.get(domain, {})

        if skill in domain_weights:
            return domain_weights[skill]

        fallback_weights = self.domain_fallback_weights.get(domain, {'default': 0.3})

        if skill_cluster and skill_cluster in fallback_weights:
            return fallback_weights[skill_cluster]

        return fallback_weights.get('default', 0.3)

    def _create_labelled_record(self, student: Dict, student_id: str, student_skills_std: set,
                                 student_interests: str, internship: Dict, internship_id: str,
                                 internship_skills_cache: Dict, internship_skills_standardized_cache: Dict,
                                 max_stipend: float, force_negative: bool = False) -> Optional[Dict]:
        internship_skills = internship_skills_cache.get(internship_id, set())
        internship_skills_std = internship_skills_standardized_cache.get(internship_id, set())

        if not internship_skills:
            return None

        student_cgpa = student['CGPA']
        student_tier = student['College Tier']
        student_region = student['Region']
        student_branch = student['Branch']
        student_location = student['Location']

        internship_domain = internship['Domain']
        internship_stipend = internship['Stipend (INR)']
        internship_duration = internship['Duration']
        internship_work_type = internship['Work Type']
        internship_location = internship['Location']

        overlap = len(student_skills_std.intersection(internship_skills_std))
        total_required = len(internship_skills_std)

        semantic_score = self._get_skill_cluster_match_score(student_skills_std, internship_skills_std)

        if total_required > 0:
            exact_ratio = overlap / total_required
            overlap_ratio = 0.6 * exact_ratio + 0.4 * semantic_score
        else:
            overlap_ratio = 0

        skill_combinations = []

        student_cluster = self._get_primary_skill_cluster(list(student_skills_std))
        intern_cluster = self._get_primary_skill_cluster(list(internship_skills_std))

        weighted_overlap = 0
        max_possible_weight = 0

        for skill in internship_skills_std:
            weight = self._get_skill_weight_with_fallback(skill, internship_domain, intern_cluster)
            max_possible_weight += weight
            if skill in student_skills_std:
                weighted_overlap += weight
                if weight > 0.7:
                    skill_combinations.append(skill)

        weighted_ratio = weighted_overlap / max_possible_weight if max_possible_weight > 0 else 0

        domain_match_score = self._get_domain_similarity_score(student_interests, internship_domain)
        domain_match = 1 if domain_match_score > 0.5 else 0

        branch_match = 0
        if student_branch in self.branch_domain_relevance:
            branch_match = self.branch_domain_relevance[student_branch].get(internship_domain, 0)

        stipend_score = internship_stipend / max_stipend if max_stipend > 0 else 0

        duration_score = min(1.0, internship_duration / 30.0)

        location_match_score = self._get_location_match_level(student_location, internship_location)

        if force_negative:
            applied = 0
            got_offer = 0
        else:
            cgpa_factor = min(1.0, student_cgpa / 8.0) if student_cgpa >= 6.0 else 0.7

            tier_factor = 1.0
            if student_tier == 'Tier-3':
                tier_factor = 0.9
            elif student_tier == 'Tier-2':
                tier_factor = 0.95
            elif student_tier == 'Tier-1':
                tier_factor = 1.0

            rural_factor = 1.0
            if internship_work_type == 'On-site' and student_region == 'Rural':
                rural_factor = 0.7

            application_score = (
                0.35 * weighted_ratio +
                0.20 * domain_match +
                0.15 * location_match_score +
                0.15 * cgpa_factor +
                0.10 * tier_factor +
                0.05 * (1 - rural_factor)
            )

            application_score = application_score * random.uniform(0.9, 1.1)
            application_prob = max(0.05, min(0.95, application_score))

            applied = 1 if random.random() < application_prob else 0

            if applied:
                success_score = (
                    0.50 * weighted_ratio +
                    0.30 * cgpa_factor +
                    0.20 * (1 if len(skill_combinations) >= 2 else 0.5)
                )

                success_prob = success_score * random.uniform(0.9, 1.1)
                success_prob = max(0.1, min(0.95, success_prob))

                got_offer = 1 if random.random() < success_prob else 0
            else:
                got_offer = 0

        return {
            'student_id': student_id,
            'internship_id': internship_id,
            'applied': applied,
            'got_offer': got_offer,
            'skill_overlap_ratio': round(overlap_ratio, 3),
            'weighted_skill_match': round(weighted_ratio, 3),
            'num_matching_skills': overlap,
            'total_required_skills': total_required,
            'has_high_value_combination': 1 if len(skill_combinations) >= 2 else 0,
            'num_high_value_skills': len(skill_combinations),
            'student_cgpa': round(student_cgpa, 2),
            'student_tier': student_tier,
            'student_region': student_region,
            'student_branch': student_branch,
            'internship_domain': internship_domain,
            'stipend_score': round(stipend_score, 3),
            'internship_duration_weeks': internship_duration,
            'duration_score': round(duration_score, 3),
            'internship_work_type': internship_work_type,
            'location_match': round(location_match_score, 2),
            'domain_match': domain_match,
            'branch_match': branch_match,
            'rural_student_on_site_penalty': 1 if (student_region == 'Rural' and internship_work_type == 'On-site') else 0
        }

    def _normalize_skills(self, skills: Any) -> str:
        if self._is_missing(skills):
            return ''

        if isinstance(skills, str):
            items = re.split(r'[,\|;]+', skills)
        elif isinstance(skills, list):
            items = skills
        else:
            items = [str(skills)]

        cleaned_skills = []
        seen_skills = set()

        for item in items:
            if self._is_missing(item):
                continue

            sub_skills = [s.strip() for s in str(item).split(',') if s.strip()]
            for skill in sub_skills:
                if self._is_missing(skill):
                    continue

                standardized_skill = self._standardize_skill(skill)
                if standardized_skill and standardized_skill.lower() not in seen_skills:
                    seen_skills.add(standardized_skill.lower())
                    cleaned_skills.append(standardized_skill)

        cleaned_skills.sort()
        return ', '.join(cleaned_skills) if cleaned_skills else ''

    def _parse_duration(self, val: Any) -> Optional[int]:
        if self._is_missing(val):
            return None

        try:
            num = int(float(val))
            if 4 <= num <= 30:
                return num
            elif 1 <= num <= 6:
                weeks = num * 4
                if 4 <= weeks <= 30:
                    return weeks
            return None
        except (ValueError, TypeError):
            pass

        str_val = str(val).lower()
        match = re.search(r'(\d+)\s*(?:month|week|mo|wk|w)', str_val)
        if match:
            num = int(match.group(1))
            if any(unit in str_val for unit in ['month', 'mo']):
                weeks = num * 4
            else:
                weeks = num

            if 4 <= weeks <= 30:
                return weeks

        match = re.search(r'(\d+)', str_val)
        if match:
            num = int(match.group(1))
            if 4 <= num <= 30:
                return num

        return None

    def _normalize_location(self, location: Any) -> str:
        if self._is_missing(location):
            return 'Unspecified'

        key = str(location).strip().lower()

        if key in self._location_cache:
            return self._location_cache[key]

        for variant, standard in self.location_map.items():
            if variant in key:
                self._location_cache[key] = standard
                return standard

        indian_cities = ['mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'pune',
                        'kolkata', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur',
                        'indore', 'thane', 'bhopal', 'visakhapatnam', 'patna', 'vadodara']

        matches = process.extract(key, indian_cities, limit=1, scorer=fuzz.token_sort_ratio)
        if matches and matches[0][1] >= 85:
            normalized = matches[0][0].capitalize()
            self._location_cache[key] = normalized
            return normalized

        if any(term in key for term in ['rural', 'village', 'gram', 'pind', 'district']):
            normalized = 'Rural India'
        else:
            normalized = 'Other City'

        self._location_cache[key] = normalized
        return normalized

    def _remove_duplicates(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        initial_count = len(df)
        df_cleaned = df.drop_duplicates(subset=[id_col], keep='first')
        duplicates_removed = initial_count - len(df_cleaned)
        if duplicates_removed > 0:
            self._record_removal_reason(f'duplicates_{id_col}')
            logger.info(f"Removed {duplicates_removed} duplicate {id_col}s")
        return df_cleaned

    def _clean_student_row(self, row: Dict) -> Optional[Dict]:
        student_id = self._clean_text(row.get('Student ID'))

        if self._is_missing(student_id):
            self._record_removal_reason('missing_student_id', 'students', 'Unknown_ID')
            return None

        branch = self._normalize_branch(row.get('Branch'))
        if not branch:
            self._record_removal_reason('missing_or_invalid_branch', 'students', student_id)
            return None
        row['Branch'] = branch

        cgpa = self._validate_number(row.get('CGPA'), 0, 10)
        if cgpa is None:
            raw_cgpa = row.get('CGPA')
            if isinstance(raw_cgpa, str) and '%' in str(raw_cgpa):
                try:
                    pct = float(str(raw_cgpa).replace('%', '').strip())
                    cgpa = pct / 10
                except ValueError:
                    pass

            if cgpa is None or cgpa < 0 or cgpa > 10:
                self._record_removal_reason('invalid_cgpa', 'students', student_id)
                return None

        if cgpa < 0 or cgpa > 10:
            self._record_removal_reason('invalid_cgpa', 'students', student_id)
            return None

        row['CGPA'] = cgpa

        skills = self._normalize_skills(row.get('Skills'))
        if not skills:
            self._record_removal_reason('missing_skills', 'students', student_id)
            return None
        row['Skills'] = skills

        interests = self._normalize_skills(row.get('Interests')) if not self._is_missing(row.get('Interests')) else ''
        row['Interests'] = interests

        tier = self._normalize_tier(row.get('College Tier'))
        if tier is None:
            self._record_removal_reason('invalid_college_tier', 'students', student_id)
            return None
        row['College Tier'] = tier

        location = self._normalize_location(row.get('Location'))
        row['Location'] = location

        if self._is_missing(row.get('Region')):
            self._record_removal_reason('missing_region', 'students', student_id)
            return None
        region = self._clean_text(row.get('Region'))
        region_lower = region.lower()
        if 'urban' in region_lower:
            region = 'Urban'
        elif 'rural' in region_lower:
            region = 'Rural'
        else:
            region = 'Urban'
        row['Region'] = region

        row['Student ID'] = student_id

        return row

    def _clean_internship_row(self, row: Dict) -> Optional[Dict]:
        internship_id = self._clean_text(row.get('Internship ID'))

        if self._is_missing(internship_id):
            self._record_removal_reason('missing_internship_id', 'internships', 'Unknown_ID')
            return None

        req_skills = self._normalize_skills(row.get('Required Skills'))
        if not req_skills:
            self._record_removal_reason('missing_required_skills', 'internships', internship_id)
            return None
        row['Required Skills'] = req_skills

        if self._is_missing(row.get('Domain')):
            self._record_removal_reason('missing_domain', 'internships', internship_id)
            return None
        domain = self._clean_text(row.get('Domain'))
        row['Domain'] = domain

        stipend = self._validate_number(row.get('Stipend (INR)'), 0, 200000, is_stipend=True)
        if stipend is None:
            stipend_raw = str(row.get('Stipend (INR)')).lower()
            if 'negotiable' in stipend_raw or 'variable' in stipend_raw or 'flexible' in stipend_raw:
                stipend = 0
            else:
                self._record_removal_reason('invalid_stipend', 'internships', internship_id)
                return None
        row['Stipend (INR)'] = int(stipend) if stipend is not None else 0

        duration = self._parse_duration(row.get('Duration'))
        if duration is None:
            self._record_removal_reason('invalid_duration', 'internships', internship_id)
            return None
        row['Duration'] = duration

        work_type = self._normalize_work_type(row.get('Work Type'))
        if work_type is None:
            self._record_removal_reason('invalid_work_type', 'internships', internship_id)
            return None
        row['Work Type'] = work_type

        row['Title'] = self._clean_text(row.get('Title')) if not self._is_missing(row.get('Title')) else 'Unnamed Internship'
        row['Company'] = self._clean_text(row.get('Company')) if not self._is_missing(row.get('Company')) else 'Unspecified Company'
        row['Location'] = self._normalize_location(row.get('Location'))
        row['Description'] = self._clean_text(row.get('Description')) if not self._is_missing(row.get('Description')) else 'No description provided'

        row['Internship ID'] = internship_id

        return row

    def load_and_clean(self, students_path: str, internships_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("=" * 80)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("=" * 80)

        try:
            students_raw = pd.read_csv(students_path, encoding='utf-8', quoting=1)
        except Exception as e:
            logger.warning(f"UTF-8 encoding failed for students: {e}. Trying latin1...")
            students_raw = pd.read_csv(students_path, encoding='latin1', quoting=1)

        try:
            internships_raw = pd.read_csv(internships_path, encoding='utf-8', quoting=1)
        except Exception as e:
            logger.warning(f"UTF-8 encoding failed for internships: {e}. Trying latin1...")
            internships_raw = pd.read_csv(internships_path, encoding='latin1', quoting=1)

        self.cleaning_stats['students_loaded'] = len(students_raw)
        self.cleaning_stats['internships_loaded'] = len(internships_raw)

        logger.info(f"Loaded {len(students_raw)} students and {len(internships_raw)} internships")

        cleaned_students = []
        for idx, row in students_raw.iterrows():
            student_id = row.get('Student ID', f'Unknown_Row_{idx}')
            try:
                cleaned = self._clean_student_row(row.to_dict())
                if cleaned is not None:
                    cleaned_students.append(cleaned)
                else:
                    self.cleaning_stats['students_removed'] += 1
            except Exception as e:
                logger.warning(f"Error cleaning student row {idx} (ID: {student_id}): {e}")
                self.cleaning_stats['students_removed'] += 1
                self._record_removal_reason('processing_error', 'students', student_id)

        cleaned_internships = []
        for idx, row in internships_raw.iterrows():
            internship_id = row.get('Internship ID', f'Unknown_Row_{idx}')
            try:
                cleaned = self._clean_internship_row(row.to_dict())
                if cleaned is not None:
                    cleaned_internships.append(cleaned)
                else:
                    self.cleaning_stats['internships_removed'] += 1
            except Exception as e:
                logger.warning(f"Error cleaning internship row {idx} (ID: {internship_id}): {e}")
                self.cleaning_stats['internships_removed'] += 1
                self._record_removal_reason('processing_error', 'internships', internship_id)

        students_df = pd.DataFrame(cleaned_students) if cleaned_students else pd.DataFrame()
        internships_df = pd.DataFrame(cleaned_internships) if cleaned_internships else pd.DataFrame()

        if not students_df.empty:
            students_df = self._remove_duplicates(students_df, 'Student ID')
        if not internships_df.empty:
            internships_df = self._remove_duplicates(internships_df, 'Internship ID')

        self.cleaning_stats['students_kept'] = len(students_df)
        self.cleaning_stats['internships_kept'] = len(internships_df)

        logger.info(f"Cleaning complete: {len(students_df)} students and {len(internships_df)} internships kept")
        logger.info("=" * 80)

        return students_df, internships_df

    def generate_labelled_dataset(self, students_df: pd.DataFrame, internships_df: pd.DataFrame,
                                output_path: str = None, seed: int = 42, sample_size: int = None) -> pd.DataFrame:
        logger.info("Generating BALANCED labelled dataset for SKILL IMPORTANCE LEARNING phase...")
        random.seed(seed)
        np.random.seed(seed)

        logger.info("Pre-processing skills and building indices...")
        student_skills_cache = {}
        internship_skills_cache = {}
        student_skills_standardized_cache = {}
        internship_skills_standardized_cache = {}
        student_interests_cache = {}

        domain_to_internships = defaultdict(list)

        cluster_to_internships = defaultdict(list)

        for idx, student in students_df.iterrows():
            skills_list = [s.strip() for s in student['Skills'].split(',') if s.strip()]
            student_skills_cache[student['Student ID']] = set(s.lower() for s in skills_list)
            student_skills_standardized_cache[student['Student ID']] = set(skills_list)

            interests = student.get('Interests', '').lower()
            if not interests or interests.strip() == '':
                inferred_interests = self._infer_interests_from_skills(skills_list)
                student_interests_cache[student['Student ID']] = inferred_interests
            else:
                student_interests_cache[student['Student ID']] = interests

        for idx, internship in internships_df.iterrows():
            skills_list = [s.strip() for s in internship['Required Skills'].split(',') if s.strip()]
            internship_skills_cache[internship['Internship ID']] = set(s.lower() for s in skills_list)
            internship_skills_standardized_cache[internship['Internship ID']] = set(skills_list)

            domain = internship['Domain']
            domain_to_internships[domain].append(internship['Internship ID'])

            primary_cluster = self._get_primary_skill_cluster(skills_list)
            if primary_cluster:
                cluster_to_internships[primary_cluster].append(internship['Internship ID'])

        logger.info(f"Built indices: {len(domain_to_internships)} domains, {len(cluster_to_internships)} skill clusters")

        labelled_data = []

        if sample_size:
            total_samples = sample_size
        else:
            total_samples = len(students_df) * 10

        positive_target = int(total_samples * 0.40)
        negative_target = int(total_samples * 0.60)

        logger.info(f"Generating {total_samples:,} samples: {positive_target:,} positive + {negative_target:,} negative")

        max_stipend = internships_df['Stipend (INR)'].max() if len(internships_df) > 0 else 1

        internships_dict = internships_df.set_index('Internship ID').to_dict('index')
        students_list = students_df.to_dict('records')

        logger.info("Generating POSITIVE samples (domain/skill matches)...")
        positive_count = 0

        for student in students_list:
            if positive_count >= positive_target:
                break

            student_id = student['Student ID']
            student_interests = student_interests_cache[student_id]
            student_skills_std = student_skills_standardized_cache[student_id]
            student_cluster = self._get_primary_skill_cluster(list(student_skills_std))

            intern_domain = self._get_domain_for_interest(student_interests)
            if intern_domain and intern_domain in domain_to_internships:
                for intern_id in domain_to_internships[intern_domain]:
                    if positive_count >= positive_target:
                        break
                    internship = internships_dict[intern_id]

                    intern_cluster = self._get_primary_skill_cluster(
                        list(internship_skills_standardized_cache[intern_id])
                    )
                    if not self._are_clusters_compatible(student_cluster, intern_cluster):
                        continue

                    record = self._create_labelled_record(
                        student, student_id, student_skills_std, student_interests,
                        internship, intern_id, internship_skills_cache, internship_skills_standardized_cache,
                        max_stipend
                    )
                    if record:
                        labelled_data.append(record)
                        positive_count += 1

            if positive_count < positive_target:
                if student_cluster and student_cluster in cluster_to_internships:
                    for intern_id in cluster_to_internships[student_cluster]:
                        if positive_count >= positive_target:
                            break
                        internship = internships_dict[intern_id]
                        record = self._create_labelled_record(
                            student, student_id, student_skills_std, student_interests,
                            internship, intern_id, internship_skills_cache, internship_skills_standardized_cache,
                            max_stipend
                        )
                        if record:
                            labelled_data.append(record)
                            positive_count += 1

            if positive_count < positive_target:
                for _ in range(5):
                    if positive_count >= positive_target:
                        break
                    random_intern = internships_df.sample(n=1).iloc[0] if len(internships_df) > 0 else None
                    if random_intern is not None:
                        intern_id = random_intern['Internship ID']
                        internship = internships_dict[intern_id]
                        intern_skills = internship_skills_standardized_cache[intern_id]

                        if student_skills_std.intersection(intern_skills):
                            record = self._create_labelled_record(
                                student, student_id, student_skills_std, student_interests,
                                internship, intern_id, internship_skills_cache, internship_skills_standardized_cache,
                                max_stipend
                            )
                            if record and record['skill_overlap_ratio'] > 0:
                                labelled_data.append(record)
                                positive_count += 1

        logger.info(f"Generated {positive_count} positive samples")

        logger.info("Generating NEGATIVE samples (non-matches)...")
        negative_count = 0

        for student in students_list:
            if negative_count >= negative_target:
                break

            student_id = student['Student ID']
            student_interests = student_interests_cache[student_id]
            student_skills_std = student_skills_standardized_cache[student_id]

            for _ in range(3):
                if negative_count >= negative_target:
                    break

                random_intern = internships_df.sample(n=1).iloc[0] if len(internships_df) > 0 else None
                if random_intern is not None:
                    intern_id = random_intern['Internship ID']
                    internship = internships_dict[intern_id]

                    record = self._create_labelled_record(
                        student, student_id, student_skills_std, student_interests,
                        internship, intern_id, internship_skills_cache, internship_skills_standardized_cache,
                        max_stipend,
                        force_negative=True
                    )
                    if record:
                        labelled_data.append(record)
                        negative_count += 1

        logger.info(f"Generated {negative_count} negative samples")
        logger.info(f"Total samples: {len(labelled_data):,}")

        labelled_df = pd.DataFrame(labelled_data)

        logger.info(f"Final dataset statistics:")
        logger.info(f"  - Total samples: {len(labelled_df):,}")
        logger.info(f"  - Applied (positive): {labelled_df['applied'].sum():,} ({labelled_df['applied'].mean()*100:.1f}%)")
        logger.info(f"  - Got offers: {labelled_df['got_offer'].sum():,} ({labelled_df['got_offer'].mean()*100:.1f}%)")
        logger.info(f"  - Skill overlap ratio (mean): {labelled_df['skill_overlap_ratio'].mean():.3f}")
        logger.info(f"  - Weighted skill match (mean): {labelled_df['weighted_skill_match'].mean():.3f}")
        logger.info(f"  - Domain match rate: {labelled_df['domain_match'].mean()*100:.1f}%")
        logger.info(f"  - Location match (mean): {labelled_df['location_match'].mean():.3f}")

        if output_path:
            labelled_df.to_csv(output_path, index=False)
            logger.info(f"Labelled dataset saved to {output_path}")

        return labelled_df

    def print_summary(self) -> None:
        print("\n" + "=" * 80)
        print("DATA CLEANING SUMMARY (ML PREPROCESSOR)")
        print("=" * 80)

        print(f"\nStudents Dataset:")
        print(f"  - Loaded:    {self.cleaning_stats['students_loaded']}")
        print(f"  - Kept:      {self.cleaning_stats['students_kept']}")
        print(f"  - Removed:   {self.cleaning_stats['students_removed']}")
        print(f"  - Retention: {self.cleaning_stats['students_kept']/self.cleaning_stats['students_loaded']*100:.1f}%")

        print(f"\nInternships Dataset:")
        print(f"  - Loaded:    {self.cleaning_stats['internships_loaded']}")
        print(f"  - Kept:      {self.cleaning_stats['internships_kept']}")
        print(f"  - Removed:   {self.cleaning_stats['internships_removed']}")
        print(f"  - Retention: {self.cleaning_stats['internships_kept']/self.cleaning_stats['internships_loaded']*100:.1f}%")

        print(f"\nDETAILED REMOVAL INFORMATION:")
        print(f"\nStudents Removed ({len(self.removed_students)} rows):")
        if self.removed_students:
            student_reason_counts = {}
            for _, reason in self.removed_students:
                student_reason_counts[reason] = student_reason_counts.get(reason, 0) + 1

            for reason, count in sorted(student_reason_counts.items()):
                print(f"  - {reason}: {count} student(s) removed")

            print(f"\nSample of removed student IDs:")
            sample_size = min(10, len(self.removed_students))
            for i in range(sample_size):
                student_id, reason = self.removed_students[i]
                print(f"  - Student ID: {student_id} - Reason: {reason}")
            if len(self.removed_students) > sample_size:
                print(f"  ... and {len(self.removed_students) - sample_size} more")
        else:
            print("  No students were removed.")

        print(f"\nInternships Removed ({len(self.removed_internships)} rows):")
        if self.removed_internships:
            internship_reason_counts = {}
            for _, reason in self.removed_internships:
                internship_reason_counts[reason] = internship_reason_counts.get(reason, 0) + 1

            for reason, count in sorted(internship_reason_counts.items()):
                print(f"  - {reason}: {count} internship(s) removed")

            print(f"\nSample of removed internship IDs:")
            sample_size = min(10, len(self.removed_internships))
            for i in range(sample_size):
                internship_id, reason = self.removed_internships[i]
                print(f"  - Internship ID: {internship_id} - Reason: {reason}")
            if len(self.removed_internships) > sample_size:
                print(f"  ... and {len(self.removed_internships) - sample_size} more")
        else:
            print("  No internships were removed.")

        if self.cleaning_stats['removal_reasons']:
            print("\nTop Overall Removal Reasons:")
            sorted_reasons = sorted(self.cleaning_stats['removal_reasons'].items(),
                                  key=lambda x: x[1], reverse=True)[:10]
            for reason, count in sorted_reasons:
                pct = count / (self.cleaning_stats['students_removed'] + self.cleaning_stats['internships_removed']) * 100
                print(f"  - {reason:40s}: {count:4d} ({pct:5.1f}%)")

        print("\n" + "=" * 80)
        print("KEY NORMALIZATION ACHIEVEMENTS")
        print("=" * 80)
        print("[OK] Duration: Strictly integer weeks (4-30 weeks range)")
        print("[OK] Work Type: Textual values with realistic variations normalized to 3 categories")
        print("[OK] Skills: 100+ tech skills standardized with proper casing preservation")
        print("[OK] Branches: Fuzzy matching handles 26+ engineering discipline variations")
        print("[OK] Locations: Indian city standardization with rural/urban detection")
        print("[OK] CGPA: Outliers capped at 10.0 with percentage format handling")
        print("=" * 80)


if __name__ == "__main__":
    preprocessor = MLPreprocessor()

    base_dir = Path(__file__).resolve().parent
    students_path = base_dir / "students_uncleaned_new_v2.csv"
    internships_path = base_dir / "internships_uncleaned_new_v2.csv"
    students_clean_path = base_dir / "students_cleaned.csv"
    internships_clean_path = base_dir / "internships_cleaned.csv"
    labelled_output_path = base_dir / "labelled_skill_importance_data.csv"

    try:
        logger.info("Step 1: Cleaning raw datasets...")
        students_clean, internships_clean = preprocessor.load_and_clean(students_path, internships_path)

        preprocessor.print_summary()

        students_clean.to_csv(students_clean_path, index=False)
        internships_clean.to_csv(internships_clean_path, index=False)
        logger.info("\n✓ Cleaned datasets saved successfully!")

        logger.info("\nStep 2: Generating labelled dataset for SKILL IMPORTANCE LEARNING phase...")
        labelled_data = preprocessor.generate_labelled_dataset(
            students_clean,
            internships_clean,
            labelled_output_path,
        )

        print("\n" + "=" * 80)
        print("SAMPLE LABELLED DATA FOR SKILL IMPORTANCE LEARNING")
        print("=" * 80)
        print(labelled_data.head(10).to_string(index=False))
        print("=" * 80)

        logger.info("\nStep 3: Analyzing skill combination importance...")
        combo_stats = labelled_data[labelled_data['applied'] == 1].groupby('has_high_value_combination').agg({
            'got_offer': ['mean', 'count'],
            'weighted_skill_match': 'mean'
        })
        print("\nSkill Combination Impact Analysis:")
        print(combo_stats)

        logger.info("✓ Preprocessing pipeline completed successfully!")
        logger.info("✓ Ready for ML model training with skill importance learning!")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Please ensure these files exist in the current directory:")
        logger.info(f"  - {students_path}")
        logger.info(f"  - {internships_path}")
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}", exc_info=True)
        raise
