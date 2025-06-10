import os
import re
import json
import numpy as np
import torch
import faiss
import pickle
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer, CrossEncoder
from google.generativeai import GenerativeModel
import google.generativeai as genai
from underthesea import word_tokenize, text_normalize, sent_tokenize
from rank_bm25 import BM25Okapi
from chat_history import ChatHistoryManager
from pydantic import BaseModel 
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter
import aioodbc

class ChatRequest(BaseModel):
    query: str
    patient_id: str
    session_id: Optional[str] = None
    include_context: bool = False
    max_results: int = 5
    show_evaluation: bool = False

class ChatResponse(BaseModel):
    response: str
    session_id: str
    context: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Union[float, str, int]]] = None
    processing_time: float
    evaluation: Optional[Dict[str, Any]] = None

class MedicalSpecialistRAGChatbot:
    def __init__(
        self, 
        json_filepath = "main_data_metadata/metadata.json",
        gemini_api_key=None,
        gemini_generation_model="gemini-1.5-flash",
        cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_dir="specialist_cache"
    ):
        self.config = {
            "json_filepath": json_filepath,
            "retrieval_top_k": 5,
            "chunk_size": 110,
            "chunk_overlap": 20,
            "confidence_threshold": 0.6,
            "cache_dir": cache_dir
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        os.makedirs(self.config["cache_dir"], exist_ok=True)

        load_dotenv()

        # Nếu không truyền gemini_api_key thì lấy từ biến môi trường
        if gemini_api_key is None:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Gemini API
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        
        # Medical domain stopwords
        self.medical_stopwords = {
            'và', 'hoặc', 'trong', 'các', 'những', 
            'là', 'của', 'có', 'để', 'với',
            'khoa', 'bệnh', 'triệu chứng', 'điều trị', 'thuốc'
        }
        
        # Domain-specific fixed words (for word tokenization)
        self.fixed_words = [
            'medicalcare', 'bệnh viện đa khoa', 'chuyên khoa', 
            'khoa nhi', 'cơ sở vật chất', 'dịch vụ'
        ]
        
        # Initialize models
        self.embedding_model = SentenceTransformer(
            'bkai-foundation-models/vietnamese-bi-encoder',
            device=self.device
        )
        self.cross_encoder = CrossEncoder(cross_encoder, device=self.device)
        self.gemini_model = GenerativeModel(gemini_generation_model)
        
        # Initialize data structures
        self.corpus_texts = []
        self.corpus_embeddings = None
        self.corpus_metadata = []
        self.bm25_index = None
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(location=":memory:")
        self.collection_name = "medical_corpus"
        
        # Store domain data with more structured format
        self.diseases = {}
        self.departments = {}
        self.specialists = {}

        self.common_symptoms = [
            'đau đầu', 'nhức đầu', 'sốt', 'sốt cao', 'sốt nhẹ', 'ớn lạnh', 'run rẩy',
            'ho', 'ho khan', 'ho có đờm', 'khó thở', 'hụt hơi', 'thở gấp',
            'đau ngực', 'tức ngực', 'nặng ngực', 'đau họng', 'viêm họng', 'rát họng',
            'buồn nôn', 'nôn', 'muốn nôn', 'tiêu chảy', 'đi lỏng', 'phân lỏng',
            'đau bụng', 'tức bụng', 'chuột rút', 'chán ăn', 'mất cảm giác thèm ăn',
            'chóng mặt', 'hoa mắt', 'choáng váng', 'mệt mỏi', 'kiệt sức', 'yếu ớt',
            'phát ban', 'nổi mẩn', 'ban đỏ', 'ngứa', 'sưng', 'phù',
            'đau khớp', 'đau cơ', 'đau xương', 'nhức mỏi', 'đau lưng',
            'sụt cân', 'giảm cân', 'gầy đi', 'mất ngủ', 'khó ngủ',
            'ra mồ hôi', 'đổ mồ hôi', 'toát mồ hôi', 'khàn giọng'
        ]

        self.stop_words = {
            'và', 'cùng', 'với', 'có', 'bị', 'cảm', 'thấy', 'giác', 'như',
            'thể', 'hiện', 'tượng', 'dấu', 'hiệu', 'triệu', 'chứng', 'ở', 'tại',
            'khu', 'vực', 'vùng', 'chỗ', 'nơi', 'còn', 'thêm', 'nữa', 'khác'
        }
        
        # Special question patterns for quick routing
        self.question_patterns = {
            "count_doctors": [r"bao\s+nhiêu\s+bác\s+sĩ", r"số\s+lượng\s+bác\s+sĩ", r"có\s+mấy\s+bác\s+sĩ"],
            "list_doctors": [r"liệt\s+kê\s+bác\s+sĩ", r"danh\s+sách\s+bác\s+sĩ", r"những\s+bác\s+sĩ\s+nào"],
            "department_info": [r"khoa\s+gì", r"thông\s+tin\s+về\s+khoa", r"khoa\s+nào\s+điều\s+trị"],
            "doctor_info": [r"thông\s+tin\s+về\s+bác\s+sĩ", r"bác\s+sĩ\s+\w+\s+là\s+ai"],
            "treatment_info": [r"điều\s+trị\s+như\s+thế\s+nào", r"cách\s+điều\s+trị", r"phương\s+pháp\s+điều\s+trị"]
        }

        self.sql_config = {
            'DRIVER': '{ODBC Driver 17 for SQL Server}',
            'SERVER': '103.109.187.223,1433',
            'DATABASE': 'AppointmentHospital',
            'UID': 'thedat',
            'PWD': 'MyPass123!',
            'TrustServerCertificate': 'yes'
        }

    async def connect_to_sql(self):
        """Kết nối đến SQL Server"""
        try:
            # Construct the connection string
            connection_string = (
                f"DRIVER={self.sql_config['DRIVER']};"
                f"SERVER={self.sql_config['SERVER']};"
                f"DATABASE={self.sql_config['DATABASE']};"
                f"UID={self.sql_config['UID']};"
                f"PWD={self.sql_config['PWD']};"
                f"TrustServerCertificate={self.sql_config['TrustServerCertificate']}"
            )
            conn = await aioodbc.connect(dsn=connection_string)
            return conn
        except Exception as e:
            logging.error(f"Error connecting to SQL Server: {e}")
            return None
    
    async def parse_doctor_name(self, full_name_with_degree):
        """
        Parse doctor name to extract degree and clean name.
        Example: "TTND. GS. TS. BS VÕ THÀNH NHÂN" -> Degree: "TTND.GS.TS.BS", Name: "VÕ THÀNH NHÂN"
        """
        if not full_name_with_degree:
            return "", ""

        # Danh sách các học vị / danh hiệu hợp lệ
        known_degrees = {"TS", "THS", "BS", "PGS", "GS", "BSCKI", "BSCKII", "BSNT", "TTND", "TTƯT", "CKII", "CKI", "CK", "CN", "CNĐK", "CNĐT", "CNĐTĐK"}

        parts = full_name_with_degree.strip().split()
        degree_parts = []
        name_start_index = 0

        for i, part in enumerate(parts):
            # Chuẩn hóa để so sánh (bỏ dấu chấm, in hoa)
            clean_part = part.replace(".", "").upper()
            if clean_part in known_degrees:
                degree_parts.append(clean_part)
            else:
                # Gặp phần không phải học vị => bắt đầu tên từ đây
                name_start_index = i
                break

        # Ghép các phần học vị bằng dấu chấm
        degree = ".".join(degree_parts)
        # Phần còn lại là tên
        name = " ".join(parts[name_start_index:]).strip()

        print(f"Parsed: '{full_name_with_degree}' -> Degree: '{degree}', Name: '{name}'")
        return degree, name

    async def get_speciality_id_by_name(self, speciality_name):
        """Lấy SpecialityId từ tên khoa/chuyên khoa"""
        conn = await self.connect_to_sql()
        if not conn:
            return None
        
        try:
            cursor = await conn.cursor()
            
            # Tìm kiếm theo tên khoa (không phân biệt hoa thường)
            query = """
            SELECT Id, SpecialityName 
            FROM Specialities 
            WHERE LOWER(LTRIM(RTRIM(SpecialityName))) LIKE ?
            """
            await cursor.execute(query, f'%{speciality_name.lower()}%')
            
            result = await cursor.fetchone()
            
            if result:
                print(f"Found speciality: ID={result.Id}, Name='{result.SpecialityName}'")
                return result.Id
            else:
                print(f"No speciality found for: '{speciality_name}'")
                return None
                
        except Exception as e:
            logging.error(f"Error getting speciality ID: {e}")
            return None
        finally:
            await conn.close()

    async def get_doctor_id_by_name(self, doctor_name_with_degree):
        """Lấy DoctorId từ tên bác sĩ có học vấn"""
        conn = await self.connect_to_sql()
        if not conn:
            return None
        
        try:
            # Parse the input to get degree and clean name
            input_degree, input_name = await self.parse_doctor_name(doctor_name_with_degree)
            
            print(f"Searching for doctor: Degree='{input_degree}', Name='{input_name}'")
            
            cursor = await conn.cursor()
            
            # Search by both FullName and Degree if both are available
            if input_degree and input_name:
                query = """
                SELECT DoctorId, FullName, Degree 
                FROM Doctors 
                WHERE LOWER(LTRIM(RTRIM(FullName))) LIKE ? 
                AND LOWER(LTRIM(RTRIM(Degree))) LIKE ?
                """
                await cursor.execute(query, f'%{input_name.lower()}%', f'%{input_degree.lower()}%')
            else:
                # Search by name only if no degree provided
                search_name = input_name if input_name else doctor_name_with_degree
                query = """
                SELECT DoctorId, FullName, Degree 
                FROM Doctors 
                WHERE LOWER(LTRIM(RTRIM(FullName))) LIKE ?
                """
                await cursor.execute(query, f'%{search_name.lower()}%')
            
            result = await cursor.fetchone()
            
            if result:
                print(f"Found doctor: ID={result.DoctorId}, FullName='{result.FullName}', Degree='{result.Degree}'")
                return result.DoctorId
            else:
                print(f"No doctor found for: '{doctor_name_with_degree}'")
                return None
                
        except Exception as e:
            logging.error(f"Error getting doctor ID: {e}")
            return None
        finally:
            await conn.close()
    
    async def preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding and retrieval"""
        text = text_normalize(text.lower())
        text = re.sub(r'[^\w\s\đĐăĂâÂêÊôÔơƠưƯ]', ' ', text)
        
        # Chỉ giữ lại từ khóa cố định thực sự cần thiết
        for fixed_word in self.fixed_words:
            text = text.replace(fixed_word, fixed_word.replace(' ', '_'))
        
        tokens = word_tokenize(text)
        # Xóa stopwords
        tokens = [token for token in tokens if token.lower() not in self.medical_stopwords]
        
        return ' '.join(tokens)
    
    async def _smart_chunk(self, text: str) -> List[str]:
        """Split text into semantic chunks preserving sentence boundaries"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = word_tokenize(sentence)
            sentence_length = len(sentence_tokens)
            
            if current_length + sentence_length > self.config["chunk_size"]:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    async def extract_specialists_info(self, specialist_data: Dict) -> Dict[str, Any]:
        """Extract and format specialist information from raw data"""
        # Extract basic information
        name = specialist_data.get("name", "")
        department = specialist_data.get("department", "")
        disease = specialist_data.get("disease", "")
        
        # Extract qualifications from name
        qualifications = ""
        clean_name = name
        if "." in name:
            parts = name.split(".")
            prefix = parts[0].strip()
            if prefix and all(c.isupper() for c in prefix if c.isalpha()):
                qualifications = prefix
                clean_name = ".".join(parts[1:]).strip()
        
        # Extract introduction text and specialties
        intro = specialist_data.get("GIỚI THIỆU", "")
        specialties = specialist_data.get("LĨNH VỰC CHUYÊN MÔN", "")
        
        # Extract education if available
        education = ""
        education_match = re.search(r'TỐT NGHIỆP\s+(.+?)(?:\.|$)', intro)
        if education_match:
            education = education_match.group(1).strip()
        
        # Extract years of experience
        experience = ""
        experience_match = re.search(r'(\d+)\s+NĂM\s+KINH NGHIỆM', intro)
        if experience_match:
            experience = f"{experience_match.group(1)} năm kinh nghiệm"
        
        # Get additional data from other specific fields
        memberships = specialist_data.get("THÀNH VIÊN TỔ CHỨC", "")
        research = specialist_data.get("CÔNG TRÌNH NGHIÊN CỨU", "")
        education_history = specialist_data.get("QUÁ TRÌNH ĐÀO TẠO", "")
        work_experience = specialist_data.get("KINH NGHIỆM CÔNG TÁC", "")
        
        # Format specialties and other list-like fields
        def format_list(text):
            if not text:
                return []
            return [item.strip() for item in re.split(r'[\n\r]+', text) if item.strip()]
        
        specialty_list = format_list(specialties)
        membership_list = format_list(memberships)
        research_list = format_list(research)
        education_list = format_list(education_history)
        work_experience_list = format_list(work_experience)
        
        # Process other fields that might be present
        additional_fields = {}
        for key, value in specialist_data.items():
            if key not in ["name", "department", "disease", "GIỚI THIỆU", "LĨNH VỰC CHUYÊN MÔN", 
                        "THÀNH VIÊN TỔ CHỨC", "CÔNG TRÌNH NGHIÊN CỨU", "QUÁ TRÌNH ĐÀO TẠO", 
                        "KINH NGHIỆM CÔNG TÁC"]:
                additional_fields[key] = value
        
        return {
            "name": clean_name,
            "qualifications": qualifications,
            "department": department,
            "disease": disease,
            "education": education,
            "experience": experience,
            "specialties": specialty_list,
            "memberships": membership_list,
            "research": research_list,
            "education_history": education_list,
            "work_experience": work_experience_list,
            "additional_fields": additional_fields,
            "full_info": specialist_data
        }
    
    async def extract_department_info(self, department_data: Dict) -> Dict[str, Any]:
        """Extract and format department information from raw data"""
        name = department_data.get("name", "")
        intro = department_data.get("THÔNG TIN GIỚI THIỆU", "")
        equipment = department_data.get("HỆ THỐNG TRANG THIẾT BỊ", "")
        techniques = department_data.get("KỸ THUẬT ĐIỀU TRỊ", "")
        
        # Extract services offered
        services = []
        if techniques:
            service_matches = re.findall(r'([A-ZĐ]+[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ,\s]+)(?:\(|\.|\,|$)', techniques)
            if service_matches:
                services = [s.strip() for s in service_matches if s.strip()]
        
        return {
            "name": name,
            "introduction": intro,
            "equipment": equipment,
            "techniques": techniques,
            "services": services,
            "full_info": department_data
        }
    
    async def load_medical_data(self):
        logging.info(f"Loading medical data from {self.config['json_filepath']}...")
        
        try:
            with open(self.config["json_filepath"], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            diseases_data = data.get("diseases", {})
            
            for disease_name, disease_data in diseases_data.items():
                self.diseases[disease_name] = disease_data
                
                # Process department info
                department_name = disease_data.get("department", "")
                department_info = disease_data.get("department_info", {})
                
                if department_name and department_info:
                    self.departments[department_name] = await self.extract_department_info(department_info)
                    # Index department information with all available keys
                    await self._index_department_data(department_name, department_info)
                
                # Process specialists info
                specialists_list = disease_data.get("specialists", [])
                for specialist in specialists_list:
                    specialist_name = specialist.get("name", "")
                    if specialist_name:
                        processed_specialist = await self.extract_specialists_info(specialist)
                        processed_specialist["disease"] = disease_name
                        self.specialists[specialist_name] = processed_specialist
                        # Index specialist information with all available keys
                        await self._index_specialist_data(specialist_name, specialist, disease_name)
                
                # Index disease metadata 
                metadata = disease_data.get("metadata", {})
                if metadata:
                    await self._index_disease_metadata(disease_name, metadata)
                
                # Index QA pairs
                await self._index_qa_pairs(disease_name, disease_data.get("qa_pairs", []))
            
            logging.info(f"Processed {len(self.corpus_texts)} chunks from JSON data")
            return len(self.corpus_texts) > 0
            
        except Exception as e:
            logging.error(f"Error processing JSON file: {e}")
            return False
    
    async def _index_disease_metadata(self, disease_name, metadata):
        """Index all fields in disease metadata"""
        for field_name, field_content in metadata.items():
            if isinstance(field_content, str) and field_content:
                chunks = await self._smart_chunk(field_content)
                
                for i, chunk in enumerate(chunks):
                    processed_chunk = await self.preprocess_text(chunk)
                    
                    if processed_chunk:
                        self.corpus_texts.append(processed_chunk)
                        self.corpus_metadata.append({
                            "disease_id": disease_name,
                            "disease_name": disease_name,
                            "section": field_name,
                            "original_text": chunk,
                            "source": f"{disease_name}.txt",
                            "type": "disease",
                            "chunk_index": i
                        })

    async def _index_qa_pairs(self, disease_name, qa_pairs):
        """Index all QA pairs for a disease"""
        for qa_pair in qa_pairs:
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            
            # Additional fields in QA pairs
            additional_fields = {}
            for key, value in qa_pair.items():
                if key not in ["question", "answer"] and value:
                    additional_fields[key] = value
            
            # Process question
            processed_question = await self.preprocess_text(question)
            if processed_question:
                self.corpus_texts.append(processed_question)
                self.corpus_metadata.append({
                    "disease_id": disease_name,
                    "disease_name": disease_name,
                    "section": "question",
                    "content": question,
                    "original_text": question,
                    "answer": answer,  # Store answer with question for direct retrieval
                    "additional_fields": additional_fields,
                    "type": "qa"
                })
            
            # Process answer
            answer_chunks = await self._smart_chunk(answer)
            
            for i, chunk in enumerate(answer_chunks):
                processed_chunk = await self.preprocess_text(chunk)
                
                if processed_chunk:
                    self.corpus_texts.append(processed_chunk)
                    self.corpus_metadata.append({
                        "disease_id": disease_name,
                        "disease_name": disease_name,
                        "section": "answer",
                        "related_question": question,
                        "original_text": chunk,
                        "additional_fields": additional_fields,
                        "type": "qa",
                        "chunk_index": i
                    })

    async def _index_department_data(self, department_name, department_info):
        """Index all fields in department data"""
        for field_name, field_content in department_info.items():
            # Skip the name field
            if field_name == "name":
                continue
                
            if isinstance(field_content, str) and field_content:
                chunks = await self._smart_chunk(field_content)
                
                for i, chunk in enumerate(chunks):
                    processed_chunk = await self.preprocess_text(chunk)
                    
                    if processed_chunk:
                        self.corpus_texts.append(processed_chunk)
                        self.corpus_metadata.append({
                            "department_name": department_name,
                            "section": field_name,
                            "original_text": chunk,
                            "type": "department",
                            "chunk_index": i
                        })

    async def _index_specialist_data(self, specialist_name, specialist_info, disease_name):
        """Index all fields in specialist data"""
        for field_name, field_content in specialist_info.items():
            # Skip the name field
            if field_name == "name" or field_name == "department" or field_name == "disease":
                continue
                
            if isinstance(field_content, str) and field_content:
                chunks = await self._smart_chunk(field_content)
                
                for i, chunk in enumerate(chunks):
                    processed_chunk = await self.preprocess_text(chunk)
                    
                    if processed_chunk:
                        self.corpus_texts.append(processed_chunk)
                        self.corpus_metadata.append({
                            "specialist_name": specialist_name,
                            "department": specialist_info.get("department", ""),
                            "disease": disease_name,
                            "section": field_name,
                            "original_text": chunk,
                            "type": "specialist",
                            "chunk_index": i
                        })
    
    async def create_indexes(self):
        """Create search indexes for retrieval"""
        # 1. Create BM25 index
        tokenized_corpus = [text.split() for text in self.corpus_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # 2. Generate embeddings
        batch_size = 32
        self.corpus_embeddings = []
        
        for i in range(0, len(self.corpus_texts), batch_size):
            batch_texts = self.corpus_texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            self.corpus_embeddings.append(batch_embeddings)
            
        self.corpus_embeddings = np.vstack(self.corpus_embeddings)
        
        # Normalize embeddings
        self.corpus_embeddings = self.corpus_embeddings / np.linalg.norm(
            self.corpus_embeddings, axis=1, keepdims=True
        )
        
        # 3. Create Qdrant collection
        vector_size = self.corpus_embeddings.shape[1]
        
        try:
            self.qdrant_client.delete_collection(self.collection_name)
        except:
            pass
            
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        # 4. Load data into Qdrant
        points = []
        for i, (embedding, metadata) in enumerate(zip(self.corpus_embeddings, self.corpus_metadata)):
            points.append(PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=metadata
            ))
            
            # Insert in batches of 100
            if len(points) >= 100 or i == len(self.corpus_embeddings) - 1:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                points = []
                
        logging.info(f"Created indexes with {len(self.corpus_texts)} entries")
    
    async def detect_question_pattern(self, query: str) -> Optional[str]:
        """Detect special question patterns for direct handling"""
        query_lower = query.lower()
        
        for pattern_type, regex_list in self.question_patterns.items():
            for regex in regex_list:
                if re.search(regex, query_lower):
                    return pattern_type
        
        return None
    
    async def determine_query_type(self, query: str) -> str:
        """Enhanced query type determination with symptom detection"""
        query_lower = query.lower()
        
        # Check for symptom patterns first
        symptom_result = await self.handle_symptom_query(query)
        if symptom_result:
            return "symptom"
        
        # Rest of the existing logic...
        specialist_patterns = [
            r'bác sĩ', r'doctor', r'bs\.', r'thầy thuốc',
            r'chuyên khoa', r'chuyên gia', r'tiến sĩ', r'giáo sư'
        ]
        
        department_patterns = [
            r'khoa', r'phòng', r'chuyên khoa', r'department',
            r'bệnh viện', r'trung tâm', r'phần'
        ]
        
        disease_patterns = [
            r'bệnh', r'hội chứng', r'triệu chứng', r'chẩn đoán',
            r'điều trị', r'thuốc', r'phương pháp'
        ]
        
        if any(re.search(pattern, query_lower) for pattern in specialist_patterns):
            return "specialist"
        elif any(re.search(pattern, query_lower) for pattern in department_patterns):
            return "department"
        elif any(re.search(pattern, query_lower) for pattern in disease_patterns):
            return "disease"
        
        return "general"

    async def predict_disease_from_symptoms(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced disease prediction from symptoms with intelligent extraction"""
        
        # Sử dụng extractor đã được cải thiện
        symptoms_found = await self.extract_symptoms_from_text(query)
        
        if not symptoms_found:
            return []
        
        print(f"Cleaned symptoms found in query: {symptoms_found}")
        
        # Calculate disease scores based on symptom matching
        disease_scores = {}
        
        for disease_name, disease_data in self.diseases.items():
            matched_symptoms = []
            total_score = 0
            
            # Get all text data from disease
            all_disease_text = []
            
            # Add description
            if 'description' in disease_data:
                all_disease_text.append(disease_data['description'])
            
            # Add QA pairs
            qa_pairs = disease_data.get('qa_pairs', [])
            for qa in qa_pairs:
                question = qa.get('question', '')
                answer = qa.get('answer', '')
                all_disease_text.extend([question, answer])
            
            # Add metadata
            metadata = disease_data.get('metadata', {})
            for key, value in metadata.items():
                if isinstance(value, str):
                    all_disease_text.append(value)
                elif isinstance(value, list):
                    all_disease_text.extend([str(v) for v in value])
            
            # Combine all text for analysis
            combined_text = ' '.join(all_disease_text).lower()
            
            # Check for symptom matches
            for symptom in symptoms_found:
                symptom_score = 0
                
                # Direct match (highest weight)
                if symptom in combined_text:
                    symptom_score += 3.0
                    matched_symptoms.append(symptom)
                
                # Partial match with keywords
                symptom_words = symptom.split()
                word_matches = 0
                for word in symptom_words:
                    if word in combined_text and len(word) > 2:  # Ignore short words
                        word_matches += 1
                
                if word_matches > 0:
                    partial_score = 2.0 * (word_matches / len(symptom_words))
                    symptom_score += partial_score
                    if symptom not in matched_symptoms:
                        matched_symptoms.append(symptom)
                
                # Semantic similarity for symptom groups
                if symptom_score < 1.0:  # Only if no strong match found
                    semantic_score = await self._calculate_semantic_similarity(symptom, combined_text)
                    symptom_score += semantic_score
                    if semantic_score > 0.5 and symptom not in matched_symptoms:
                        matched_symptoms.append(symptom)
                
                total_score += symptom_score
            
            # Bonus for multiple symptom matches
            if len(matched_symptoms) > 1:
                total_score *= (1 + 0.3 * (len(matched_symptoms) - 1))
            
            # Normalize score based on number of input symptoms
            if len(symptoms_found) > 0:
                normalized_score = total_score / len(symptoms_found)
                
                # Only include diseases with meaningful matches
                if normalized_score > 0.3:  # Lowered threshold for better sensitivity
                    disease_scores[disease_name] = {
                        'score': normalized_score,
                        'matched_symptoms': matched_symptoms,
                        'raw_score': total_score,
                        'disease_data': disease_data
                    }
        
        # Sort by score and return top matches
        sorted_diseases = sorted(
            disease_scores.items(), 
            key=lambda x: (x[1]['score'], len(x[1]['matched_symptoms'])), 
            reverse=True
        )
        
        # Return top 5 matches with detailed information
        result = []
        for disease_name, score_data in sorted_diseases[:5]:
            result.append({
                'disease_name': disease_name,
                'score': score_data['score'],
                'matched_symptoms': score_data['matched_symptoms'],
                'disease_data': score_data['disease_data']
            })
        
        print(f"Predicted diseases: {[(d['disease_name'], d['score']) for d in result]}")
        return result

    async def _calculate_semantic_similarity(self, symptom: str, disease_text: str) -> float:
        """Calculate semantic similarity between symptom and disease text"""
        score = 0
        
        # Group related symptoms
        symptom_groups = {
            'respiratory': ['ho', 'khó thở', 'hụt hơi', 'thở gấp', 'đau ngực', 'viêm họng'],
            'fever': ['sốt', 'sốt cao', 'ớn lạnh', 'run rẩy'],
            'digestive': ['buồn nôn', 'nôn', 'tiêu chảy', 'đau bụng', 'chán ăn'],
            'neurological': ['đau đầu', 'chóng mặt', 'hoa mắt', 'mệt mỏi'],
            'skin': ['ngứa', 'sưng', 'phát ban', 'nổi mẩn'],
            'pain': ['đau khớp', 'đau cơ', 'đau lưng', 'nhức mỏi']
        }
        
        # Find symptom group
        symptom_group = None
        for group, symptoms in symptom_groups.items():
            if any(s in symptom for s in symptoms):
                symptom_group = group
                break
        
        if symptom_group:
            # Check if disease text contains related terms
            group_keywords = {
                'respiratory': ['hô hấp', 'phổi', 'viêm phế quản', 'hen suyễn', 'cảm cúm', 'viêm họng'],
                'fever': ['nhiệt độ', 'nhiễm trùng', 'viêm', 'virus', 'vi khuẩn', 'infection'],
                'digestive': ['tiêu hóa', 'dạ dày', 'ruột', 'gan', 'mật', 'gastro'],
                'neurological': ['thần kinh', 'não', 'đầu', 'mạch máu não', 'neuro'],
                'skin': ['da', 'derma', 'viêm da', 'dị ứng da'],
                'pain': ['cơ xương khớp', 'xương', 'khớp', 'cơ', 'musculo']
            }
            
            keywords = group_keywords.get(symptom_group, [])
            for keyword in keywords:
                if keyword in disease_text:
                    score += 1.0
                    break
        
        return min(score, 2.0)  # Cap at 2.0
    
    async def _get_departments_for_diseases(self, disease_list: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Get all departments that can treat the predicted diseases"""
        relevant_departments = {}
        
        for disease_info in disease_list:
            disease_data = disease_info['disease_data']
            department_name = disease_data.get("department", "")
            
            if department_name and department_name in self.departments:
                if department_name not in relevant_departments:
                    relevant_departments[department_name] = {
                        'department_data': self.departments[department_name],
                        'diseases_treated': [],
                        'department_id': await self.get_speciality_id_by_name(department_name)
                    }
                relevant_departments[department_name]['diseases_treated'].append(disease_info['disease_name'])
        
        return relevant_departments

    async def _get_doctors_for_diseases(self, disease_list: List[Dict[str, Any]]) -> List[Dict]:
        """Get all doctors who can treat the predicted diseases with enhanced information"""
        all_doctors = {}
        
        # Collect doctors from disease data
        for disease_info in disease_list:
            disease_data = disease_info['disease_data']
            specialists = disease_data.get("specialists", [])
            
            for specialist in specialists:
                specialist_name = specialist.get("name", "")
                if specialist_name and specialist_name in self.specialists:
                    if specialist_name not in all_doctors:
                        specialist_info = self.specialists[specialist_name]
                        all_doctors[specialist_name] = {
                            'name': specialist_name,
                            'qualifications': specialist_info.get("qualifications", ""),
                            'experience': specialist_info.get("experience", ""),
                            'department': specialist_info.get("department", ""),
                            'specialties': specialist_info.get("specialties", []),
                            'doctor_id': await self.get_doctor_id_by_name(specialist_name),
                            'diseases_can_treat': [],
                            'relevance_score': 0
                        }
                    
                    all_doctors[specialist_name]['diseases_can_treat'].append(disease_info['disease_name'])
                    all_doctors[specialist_name]['relevance_score'] += disease_info['score']
        
        # Also add doctors from relevant departments
        relevant_departments = await self._get_departments_for_diseases(disease_list)
        for dept_name in relevant_departments.keys():
            dept_doctors = [s for s_name, s_data in self.specialists.items() 
                        if s_data.get("department") == dept_name]
            
            for doctor_data in dept_doctors:
                doctor_name = doctor_data.get("name", "")
                if doctor_name and doctor_name not in all_doctors:
                    all_doctors[doctor_name] = {
                        'name': doctor_name,
                        'qualifications': doctor_data.get("qualifications", ""),
                        'experience': doctor_data.get("experience", ""),
                        'department': doctor_data.get("department", ""),
                        'specialties': doctor_data.get("specialties", []),
                        'doctor_id': await self.get_doctor_id_by_name(doctor_name),
                        'diseases_can_treat': [d['disease_name'] for d in disease_list if d['disease_data'].get('department') == dept_name],
                        'relevance_score': sum([d['score'] for d in disease_list if d['disease_data'].get('department') == dept_name]) * 0.7  # Lower score for department-based matching
                    }
        
        # Convert to list and sort by relevance
        doctor_list = list(all_doctors.values())
        doctor_list.sort(key=lambda x: (len(x['diseases_can_treat']), x['relevance_score']), reverse=True)
        
        return doctor_list
    
    async def transform_query(self, query: str) -> str:
        """Transform query for retrieval"""
        return await self.preprocess_text(query)
    
    async def hybrid_search(self, query: str, top_k: int = None, filter_type=None) -> List[Dict[str, Any]]:
        """Asynchronous hybrid search combining dense and sparse retrieval with reranking"""
        if top_k is None:
            top_k = self.config["retrieval_top_k"]

        processed_query = await self.transform_query(query)
        query_tokens = processed_query.split()

        # Thực hiện đồng thời dense và sparse retrieval
        dense_task = asyncio.create_task(self._perform_dense_retrieval(processed_query, top_k, filter_type))
        sparse_task = asyncio.create_task(self._perform_sparse_retrieval(query_tokens, top_k))

        dense_indices, bm25_indices = await asyncio.gather(dense_task, sparse_task)

        # Kết hợp kết quả và loại bỏ trùng lặp
        candidates = list(set(dense_indices + bm25_indices))

        # Reranking với threshold cao hơn để lọc kết quả chính xác
        results = await self._rerank_candidates(processed_query, candidates, filter_type, top_k)

        return results

    async def _perform_dense_retrieval(self, processed_query, top_k, filter_type):
        """Perform dense retrieval using embeddings"""
        query_embedding = self.embedding_model.encode([processed_query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        search_filter = None
        if filter_type:
            search_filter = Filter(
                must=[{"key": "type", "match": {"value": filter_type}}]
            )
            
        dense_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k * 2,
            query_filter=search_filter
        )
        
        return [hit.id for hit in dense_results]

    async def _perform_sparse_retrieval(self, query_tokens, top_k):
        """Perform sparse retrieval using BM25"""
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        return np.argsort(bm25_scores)[::-1][:top_k * 2].tolist()

    async def _rerank_candidates(self, processed_query, candidates, filter_type, top_k):
        """Rerank candidates using cross-encoder with higher threshold"""
        reranked_candidates = []
        for idx in candidates:
            if filter_type and self.corpus_metadata[idx].get("type") != filter_type:
                continue
                
            score = self.cross_encoder.predict([(processed_query, self.corpus_texts[idx])])[0]
            reranked_candidates.append((idx, score))
        
        # Sort by score
        reranked_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results with higher threshold for accuracy
        results = []
        threshold = max(self.config["confidence_threshold"], 0.75)  # Tăng threshold để có kết quả chính xác hơn
        
        for idx, score in reranked_candidates[:top_k]:
            if score > threshold:
                metadata = self.corpus_metadata[idx].copy()
                metadata["original_text"] = metadata.get("original_text", self.corpus_texts[idx])
                metadata["score"] = float(score)
                
                results.append({
                    "text": self.corpus_texts[idx],
                    "metadata": metadata,
                    "score": float(score)
                })
        
        return results
    
    async def auto_merge_chunks(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge adjacent chunks from the same source for better context"""
        if not results:
            return results
            
        # Group results by source and type
        grouped_results = {}
        for result in results:
            metadata = result["metadata"]
            
            # Tạo group_key chính xác để merge đúng chunks
            group_key = None
            if metadata.get("type") == "disease":
                group_key = f"disease_{metadata.get('disease_id', metadata.get('disease_name', ''))}"
            elif metadata.get("type") == "qa":
                # Đối với QA, group theo question để đảm bảo answer đúng
                if metadata.get("section") == "answer":
                    group_key = f"qa_{metadata.get('related_question', metadata.get('question', ''))}"
                else:
                    group_key = f"qa_{metadata.get('question', '')}"
            elif metadata.get("type") == "specialist":
                group_key = f"specialist_{metadata.get('specialist_name', '')}"
            elif metadata.get("type") == "department":
                group_key = f"department_{metadata.get('department_name', '')}_{metadata.get('section', '')}"
                
            if group_key:
                if group_key not in grouped_results:
                    grouped_results[group_key] = []
                grouped_results[group_key].append(result)
        
        # Merge chunks within each group
        merged_results = []
        for group_results in grouped_results.values():
            # Sort by chunk_index if available
            group_results.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
            
            # Nếu chỉ có một đoạn hoặc không có chỉ mục đoạn, giữ nguyên
            if len(group_results) <= 1 or "chunk_index" not in group_results[0]["metadata"]:
                merged_results.extend(group_results)
                continue
                
            # Merge adjacent chunks
            current_merged = group_results[0].copy()
            current_text = current_merged["metadata"]["original_text"]
            max_score = current_merged["score"]
            
            for i in range(1, len(group_results)):
                # Nếu các đoạn liền kề, hợp nhất chúng
                if (group_results[i]["metadata"].get("chunk_index", 0) == 
                    group_results[i-1]["metadata"].get("chunk_index", 0) + 1):
                    current_text += " " + group_results[i]["metadata"]["original_text"]
                    max_score = max(max_score, group_results[i]["score"])
                else:
                    # Hoàn thành đoạn đã hợp nhất và bắt đầu đoạn mới
                    current_merged["metadata"]["original_text"] = current_text
                    current_merged["score"] = max_score
                    merged_results.append(current_merged)
                    
                    current_merged = group_results[i].copy()
                    current_text = current_merged["metadata"]["original_text"]
                    max_score = current_merged["score"]
            
            # Thêm đoạn hợp nhất cuối cùng
            current_merged["metadata"]["original_text"] = current_text
            current_merged["score"] = max_score
            merged_results.append(current_merged)
        
        # Sắp xếp lại theo điểm số
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        return merged_results[:self.config["retrieval_top_k"]]
    
    async def is_relevant_query(self, query: str) -> bool:
        """Check if the query is relevant to the medical domain - optimized"""
        relevant_keywords = [
            'bệnh', 'chàm', 'eczema', 'da liễu', 'viêm da', 'dị ứng',
            'triệu chứng', 'điều trị', 'viêm', 'kích ứng', 'dị ứng',
            'răng', 'nha khoa', 'khám bệnh', 'bác sĩ', 'chuyên khoa',
            'phẫu thuật', 'khoa', 'viện', 'medicalcare', 'bệnh viện',
            'phòng khám', 'chữa', 'cách', 'làm sao', 'nha sĩ', 'giun',
            'chẩn đoán', 'xét nghiệm'
        ]
        
        processed_query = query.lower()
        
        # Quick keyword check first
        for keyword in relevant_keywords:
            if keyword in processed_query:
                return True
        
        # Check disease names
        for disease_name in self.diseases.keys():
            if disease_name.lower() in processed_query:
                return True
                
        # Check specialist names
        for specialist_name in self.specialists.keys():
            if specialist_name.lower() in processed_query:
                return True
        
        # Only do search if quick checks fail
        results = await self.hybrid_search(query, top_k=1)
        if results and results[0]["score"] > 0.8:  # Tăng threshold
            return True
            
        return False
    
    async def handle_list_doctors_query(self, query: str) -> Optional[str]:
        """Handle questions requesting a list of doctors with direct links"""
        query_lower = query.lower()
        
        # Check for specific department mention
        for department_name in self.departments.keys():
            if department_name.lower() in query_lower:
                # List specialists in the department
                specialists = [s for s in self.specialists.values() 
                            if s.get("department") == department_name]
                
                if specialists:
                    response = f"Danh sách bác sĩ thuộc Khoa {department_name}:\n\n"
                    
                    # Get department ID for department link
                    department_id = await self.get_speciality_id_by_name(department_name)
                    if department_id:
                        response += f"🏥 <a href='/Patient/DetailSpecialities/{department_id}' class='btn btn-outline-primary btn-hover-fill' style='margin-bottom: 15px; display: inline-block;'>Xem thông tin chi tiết Khoa {department_name}</a>\n\n"
                    
                    for i, specialist in enumerate(specialists, 1):
                        name = specialist.get("name", "")
                        qualifications = specialist.get("qualifications", "")
                        experience = specialist.get("experience", "")
                        
                        # Get doctor ID from database
                        doctor_id = await self.get_doctor_id_by_name(name)
                        
                        response += f"{i}. "
                        if qualifications:
                            response += f"{qualifications} "
                        response += f"{name}"
                        if experience:
                            response += f" ({experience})"
                        
                        # Add direct link if doctor ID is found
                        if doctor_id:
                            response += f"\n   👨‍⚕️ <a href='/Patient/DetailDoctor/{doctor_id}' class='btn btn-outline-primary btn-hover-fill'>Xem chi tiết bác sĩ</a>"
                        
                        response += "\n\n"
                    
                    return response
                else:
                    return f"Hiện tại không có thông tin về các bác sĩ trong Khoa {department_name}."
        
        # If no specific department is mentioned, list all doctors
        response = "Danh sách các bác sĩ chuyên khoa tại bệnh viện:\n\n"
        sorted_specialists = sorted(self.specialists.values(), key=lambda x: x.get("department", ""))
        
        for i, specialist in enumerate(sorted_specialists, 1):
            name = specialist.get("name", "")
            qualifications = specialist.get("qualifications", "")
            department = specialist.get("department", "")
            
            # Get doctor ID from database
            doctor_id = await self.get_doctor_id_by_name(name)
            
            response += f"{i}. "
            if qualifications:
                response += f"{qualifications} "
            response += name
            if department:
                response += f" - Khoa {department}"
            
            # Add direct link if doctor ID is found
            if doctor_id:
                response += f"\n   👨‍⚕️ <a href='/Patient/DetailDoctor/{doctor_id}' class='btn btn-outline-primary btn-hover-fill'>Xem chi tiết bác sĩ</a>"
            
            response += "\n\n"
        
        return response
    
    async def handle_doctor_info_query(self, query: str) -> Optional[str]:
        """Handle questions about doctor information with direct link"""
        query_lower = query.lower()
        # Check for doctor name mention
        for specialist_name, specialist_data in self.specialists.items():
            if specialist_name.lower() in query_lower:
                name = specialist_data.get("name", "")
                qualifications = specialist_data.get("qualifications", "")
                department = specialist_data.get("department", "")
                education = specialist_data.get("education", "")
                experience = specialist_data.get("experience", "")
                specialties = specialist_data.get("specialties", [])
                
                print(f"Processing doctor info for: {name}")
                
                response = ""
                if qualifications:
                    response += f"{qualifications} "
                response += f"{name}"
                if department:
                    response += f" - Khoa {department}\n\n"
                else:
                    response += "\n\n"
                
                # Only show information if it exists
                if education:
                    response += f"Đào tạo: {education}\n"
                
                if experience:
                    response += f"Kinh nghiệm: {experience}\n"
                
                if specialties:
                    response += "Lĩnh vực chuyên môn:\n"
                    for specialty in specialties:
                        response += f"- {specialty}\n"
                
                # Get doctor ID and add direct link
                doctor_id = await self.get_doctor_id_by_name(name)
                if doctor_id:
                    response += f"\n👨‍⚕️ <a href='/Patient/DetailDoctor/{doctor_id}' class='btn btn-outline-primary btn-hover-fill'>Xem chi tiết đầy đủ</a>"
                
                # Get department ID and add department link
                if department:
                    department_id = await self.get_speciality_id_by_name(department)
                    if department_id:
                        response += f"\n🏥 <a href='/Patient/DetailSpecialities/{department_id}' class='btn btn-outline-primary btn-hover-fill'>Xem thông tin Khoa {department}</a>"
                
                return response
        
        # If no specific doctor is found
        return None
    
    async def handle_specialized_queries(self, query: str) -> Optional[str]:
        """Handle specialized query patterns with direct responses"""
        pattern_type = await self.detect_question_pattern(query)
        
        if pattern_type == "list_doctors":
            return await self.handle_list_doctors_query(query)
        elif pattern_type == "doctor_info":
            return await self.handle_doctor_info_query(query)
        
        return None
    
    async def generate_system_prompt(self, query_type: str) -> str:
        """Generate system prompt based on query type"""
        general_instructions = """
        Bạn là trợ lý y tế thông minh của bệnh viện chuyên khoa, chuyên cung cấp thông tin chính xác và đáng tin cậy về các vấn đề sức khỏe, bệnh lý, chuyên khoa và bác sĩ của bệnh viện.
        
        Nhiệm vụ của bạn:
        1. Cung cấp thông tin chính xác, rõ ràng và dễ hiểu, chỉ trả lời những câu hỏi liên quan đến dữ liệu có trong hệ thống
        2. Trả lời dựa trên dữ liệu y khoa từ các tài liệu đã được cung cấp, không bịa đặt thông tin
        3. Sử dụng ngôn ngữ thân thiện, chuyên nghiệp và dễ hiểu
        4. Không chẩn đoán bệnh, chỉ cung cấp thông tin tham khảo
        5. Hướng dẫn bệnh nhân tới các chuyên khoa hoặc bác sĩ phù hợp CHÍNH XÁC với bệnh lý
        6. Sử dụng đầy đủ thông tin đã được cung cấp trong context, không nói thiếu
        7. Chỉ giới thiệu bác sĩ và khoa có liên quan trực tiếp đến bệnh lý được hỏi
        8. Không cần hiển thị thông tin đó không có sẵn, chỉ hiển thị những gì đã được cung cấp

        Lưu ý:
        - Luôn khuyến khích người dùng tham khảo ý kiến của bác sĩ
        - Không đưa ra những lời khẳng định tuyệt đối về các phương pháp điều trị
        - Thể hiện sự đồng cảm và hiểu biết về các vấn đề sức khỏe
        - Chỉ hiển thị thông tin có sẵn, ẩn hoàn toàn những thông tin không có(ko cần hiển thị là thông tin không có)
        - Không sử dụng ngôn ngữ chuyên ngành quá phức tạp, hãy giải thích rõ ràng
        """
        
        if query_type == "specialist":
            return general_instructions + """
            Bạn đang cung cấp thông tin về bác sĩ chuyên khoa. Hãy tập trung vào:
            - Chuyên môn và lĩnh vực chuyên sâu của bác sĩ (chỉ hiển thị nếu có)
            - Kinh nghiệm và thành tựu chuyên môn (chỉ hiển thị nếu có)
            - Khoa/phòng khám nơi bác sĩ làm việc (chỉ hiển thị nếu có)
            - Các bệnh lý mà bác sĩ này có thể điều trị
            - TUYỆT ĐỐI KHÔNG nói "chưa được cung cấp" hoặc "không có thông tin"
            - Chỉ hiển thị những thông tin có sẵn trong dữ liệu
            """
        elif query_type == "department":
            return general_instructions + """
            Bạn đang cung cấp thông tin về khoa phòng của bệnh viện. Hãy tập trung vào:
            - Chức năng và nhiệm vụ của khoa
            - Các bệnh lý được điều trị tại khoa
            - Trang thiết bị và công nghệ nổi bật (chỉ hiển thị nếu có)
            - Đội ngũ y bác sĩ của khoa có liên quan
            """
        elif query_type == "disease":
            return general_instructions + """
            Bạn đang cung cấp thông tin về bệnh lý. Hãy tập trung vào:
            - Thông tin y khoa về bệnh (nguyên nhân, triệu chứng, điều trị)
            - Các biện pháp phòng ngừa (chỉ hiển thị nếu có)
            - Bác sĩ chuyên khoa có thể điều trị CHÍNH XÁC bệnh này
            - Khoa phòng liên quan TRỰC TIẾP đến bệnh lý
            - Sử dụng thông tin cụ thể từ dữ liệu thay vì nói chung chung
            - CHỈ giới thiệu bác sĩ và khoa có chuyên môn phù hợp với bệnh được hỏi
            """
        else:
            return general_instructions

    async def _format_context_entry(self, metadata, original_text, index):
        """Format a single context entry based on its type - improved accuracy"""
        if metadata.get("type") == "qa":
            if metadata.get("section") == "question":
                answer = metadata.get("answer", "")
                return f"Câu hỏi: {original_text}\nTrả lời: {answer}\n\n"
            elif metadata.get("section") == "answer":
                # Đảm bảo trích xuất đúng question cho answer
                related_question = metadata.get("related_question", metadata.get("question", ""))
                if related_question:
                    return f"Câu hỏi: {related_question}\nTrả lời: {original_text}\n\n"
                else:
                    return f"Thông tin: {original_text}\n\n"
            else:
                return f"Thông tin Q&A: {original_text}\n\n"
        else:
            source_info = ""
            if metadata.get("type") == "disease":
                source_info = f"Thông tin về bệnh {metadata.get('disease_name', '')}"
            elif metadata.get("type") == "specialist":
                source_info = f"Thông tin về bác sĩ {metadata.get('specialist_name', '')}"
            elif metadata.get("type") == "department":
                source_info = f"Thông tin về Khoa {metadata.get('department_name', '')}"
            
            if source_info:
                return f"{source_info}: {original_text}\n\n"
            else:
                return f"Thông tin {index}: {original_text}\n\n"
    
    
    async def get_disease_name_from_query(self, query: str) -> Optional[str]:
        """Extract disease name from query if present - improved matching"""
        query_lower = query.lower()
        
        # Tìm kiếm chính xác tên bệnh trong query
        best_match = None
        max_match_length = 0
        
        for disease_name in self.diseases.keys():
            disease_lower = disease_name.lower()
            if disease_lower in query_lower:
                # Ưu tiên match dài hơn (cụ thể hơn)
                if len(disease_lower) > max_match_length:
                    best_match = disease_name
                    max_match_length = len(disease_lower)
        
        return best_match
    
    async def get_specialist_name_from_query(self, query: str) -> Optional[str]:
        """Extract specialist name from query if present"""
        query_lower = query.lower()
        
        for specialist_name in self.specialists.keys():
            if specialist_name.lower() in query_lower:
                return specialist_name
                
        return None
        
    async def get_department_name_from_query(self, query: str) -> Optional[str]:
        """Extract department name from query if present"""
        query_lower = query.lower()
        
        for department_name in self.departments.keys():
            if department_name.lower() in query_lower:
                return department_name
                
        return None
    
    async def generate_response(self, query: str, results: List[Dict[str, Any]], query_type: str) -> str:
        """Generate response using Gemini model with enhanced symptom handling"""
        # Handle symptom queries with comprehensive disease prediction
        if query_type == "symptom":
            predicted_diseases = await self.predict_disease_from_symptoms(query)
            if predicted_diseases:
                return await self.generate_symptom_response(query, predicted_diseases)
            # If no diseases predicted, fall through to regular processing
        
        # Regular processing for other query types
        system_prompt = await self.generate_system_prompt(query_type)
        context = await self.format_retrieval_context(results)
        
        # Enrich context with entity-specific information
        disease_name = await self.get_disease_name_from_query(query)
        specialist_name = await self.get_specialist_name_from_query(query)
        department_name = await self.get_department_name_from_query(query)
        
        if disease_name and disease_name in self.diseases:
            disease_data = self.diseases[disease_name]
            context += f"\nThông tin chi tiết về {disease_name}:\n"
            
            # Extract information from disease metadata - only show if exists
            for key, value in disease_data.items():
                if key == "name":
                    continue  # Skip disease name as it's already mentioned
                elif key == "metadata" and isinstance(value, dict):
                    # Process metadata - only show non-empty values
                    for meta_key, meta_value in value.items():
                        if meta_value and meta_key != "description":
                            context += f"- {meta_key.replace('_', ' ').title()}: {meta_value}\n"
                elif key == "qa_pairs" and isinstance(value, list):
                    # Process Q&A pairs - extract accurate medical information
                    medical_info = {}
                    
                    for qa in value:
                        if isinstance(qa, dict):
                            question = qa.get("question", "").lower()
                            answer = qa.get("answer", "")
                            
                            if not answer:
                                continue
                            
                            # Categorize medical information based on question
                            if any(keyword in question for keyword in ["là gì", "định nghĩa", "khái niệm"]):
                                medical_info["definition"] = answer
                            elif any(keyword in question for keyword in ["nguyên nhân", "tại sao", "do đâu", "gây ra"]):
                                medical_info["causes"] = answer
                            elif any(keyword in question for keyword in ["triệu chứng", "dấu hiệu", "biểu hiện", "cảm giác"]):
                                medical_info["symptoms"] = answer
                            elif any(keyword in question for keyword in ["chẩn đoán", "xét nghiệm", "kiểm tra", "phát hiện"]):
                                medical_info["diagnosis"] = answer
                            elif any(keyword in question for keyword in ["điều trị", "chữa", "thuốc", "phương pháp", "cách chữa"]):
                                medical_info["treatment"] = answer
                            elif any(keyword in question for keyword in ["phòng ngừa", "tránh", "ngăn chặn", "dự phòng"]):
                                medical_info["prevention"] = answer
                            elif any(keyword in question for keyword in ["biến chứng", "nguy hiểm", "hệ quả", "tác hại"]):
                                medical_info["complications"] = answer
                            elif any(keyword in question for keyword in ["chế độ", "kiêng", "ăn uống", "sinh hoạt"]):
                                medical_info["lifestyle"] = answer
                            elif any(keyword in question for keyword in ["theo dõi", "tái khám", "kiểm soát", "quan sát"]):
                                medical_info["follow_up"] = answer
                            else:
                                # Other information
                                if "other_info" not in medical_info:
                                    medical_info["other_info"] = []
                                medical_info["other_info"].append(f"{qa.get('question', '')}: {answer}")
                    
                    # Display medical information in logical order - only if exists
                    info_order = [
                        ("definition", "Định nghĩa"),
                        ("causes", "Nguyên nhân"),
                        ("symptoms", "Triệu chứng"),
                        ("diagnosis", "Chẩn đoán"),
                        ("treatment", "Điều trị"),
                        ("prevention", "Phòng ngừa"),
                        ("complications", "Biến chứng"),
                        ("lifestyle", "Chế độ sinh hoạt"),
                        ("follow_up", "Theo dõi")
                    ]
                    
                    for key, label in info_order:
                        if key in medical_info and medical_info[key]:
                            context += f"- {label}: {medical_info[key]}\n"
                    
                    # Add other information if exists
                    if "other_info" in medical_info and medical_info["other_info"]:
                        context += "- Thông tin bổ sung:\n"
                        for info in medical_info["other_info"]:
                            context += f"  • {info}\n"
                elif key == "specialists" and isinstance(value, list):
                    # Process specialist list - only show if exists
                    context += "- Bác sĩ chuyên khoa điều trị:\n"
                    for specialist in value:
                        if isinstance(specialist, dict):
                            spec_name = specialist.get("name", "")
                            if spec_name and spec_name in self.specialists:
                                specialist_info = self.specialists[spec_name]
                                qualifications = specialist_info.get("qualifications", "")
                                experience = specialist_info.get("experience", "")
                                department = specialist_info.get("department", "")
                                
                                # Display doctor information - only if exists
                                display_name = f"{qualifications} {spec_name}" if qualifications else spec_name
                                context += f"  + {display_name}\n"
                                
                                if experience:
                                    context += f"    Kinh nghiệm: {experience}\n"
                                if department:
                                    context += f"    Khoa: {department}\n"
                                
                                # Add doctor link
                                doctor_id = await self.get_doctor_id_by_name(spec_name)
                                if doctor_id:
                                    context += f"    👨‍⚕️ <a href='/Patient/DetailDoctor/{doctor_id}' class='btn btn-outline-primary btn-hover-fill'>Xem chi tiết bác sĩ</a>\n"
                                context += "\n"
                elif key == "department":
                    # Process department information - only if exists
                    if isinstance(value, str) and value:
                        department_id = await self.get_speciality_id_by_name(value)
                        context += f"- Điều trị tại: Khoa {value}\n"
                        if department_id:
                            context += f"  🏥 <a href='/Patient/DetailSpecialities/{department_id}' class='btn btn-outline-primary btn-hover-fill'>Xem thông tin Khoa {value}</a>\n"
                    elif isinstance(value, dict):
                        dept_name = value.get("name", "")
                        if dept_name:
                            department_id = await self.get_speciality_id_by_name(dept_name)
                            context += f"- Điều trị tại: {dept_name}\n"
                            if department_id:
                                context += f"  🏥 <a href='/Patient/DetailSpecialities/{department_id}' class='btn btn-outline-primary btn-hover-fill'>Xem thông tin {dept_name}</a>\n"
                            
                            # Add detailed department information if available
                            for dept_key, dept_value in value.items():
                                if dept_key != "name" and dept_value:
                                    if isinstance(dept_value, list):
                                        context += f"  - {dept_key.replace('_', ' ').title()}: {', '.join(dept_value)}\n"
                                    else:
                                        context += f"  - {dept_key.replace('_', ' ').title()}: {dept_value}\n"
                elif isinstance(value, str) and value:
                    # Process other string information - only if exists
                    context += f"- {key.replace('_', ' ').title()}: {value}\n"
                elif isinstance(value, list) and value:
                    # Process lists - only if exists
                    context += f"- {key.replace('_', ' ').title()}: {', '.join(str(v) for v in value)}\n"
                            
        if specialist_name and specialist_name in self.specialists:
            specialist_data = self.specialists[specialist_name]
            context += f"\nThông tin bổ sung về bác sĩ {specialist_name}:\n"
            
            # Add specialist data - only if exists
            qualifications = specialist_data.get("qualifications", "")
            if qualifications:
                context += f"- Trình độ: {qualifications}\n"
                
            experience = specialist_data.get("experience", "")
            if experience:
                context += f"- Kinh nghiệm: {experience}\n"
                
            department = specialist_data.get("department", "")
            if department:
                context += f"- Khoa: {department}\n"
                
            specialties = specialist_data.get("specialties", [])
            if specialties:
                if isinstance(specialties, list):
                    context += f"- Chuyên môn: {', '.join(specialties)}\n"
                else:
                    context += f"- Chuyên môn: {specialties}\n"
            
            education = specialist_data.get("education", "")
            if education:
                context += f"- Đào tạo: {education}\n"
            
            # Add other data - only if exists
            for key, value in specialist_data.items():
                if key not in ["name", "qualifications", "experience", "department", "specialties", "education"] and value:
                    if isinstance(value, list):
                        context += f"- {key.replace('_', ' ').title()}: {', '.join(str(v) for v in value)}\n"
                    else:
                        context += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            # Add doctor link at the end
            doctor_id = await self.get_doctor_id_by_name(specialist_name)
            if doctor_id:
                context += f"\n👨‍⚕️ <a href='/Patient/DetailDoctor/{doctor_id}' class='doctor-detail-link btn btn-outline-primary btn-hover-fill'>Xem hồ sơ đầy đủ của bác sĩ</a>\n"
            
            # Add department link if available
            if department:
                department_id = await self.get_speciality_id_by_name(department)
                if department_id:
                    context += f"🏥 <a href='/Patient/DetailSpecialities/{department_id}' class='btn btn-outline-primary btn-hover-fill'>Xem thông tin Khoa {department}</a>\n"
        
        if department_name and department_name in self.departments:
            department_data = self.departments[department_name]
            context += f"\nThông tin bổ sung về Khoa {department_name}:\n"
            
            # Add department link first
            department_id = await self.get_speciality_id_by_name(department_name)
            if department_id:
                context += f"🏥 <a href='/Patient/DetailSpecialities/{department_id}' class='btn btn-outline-primary btn-hover-fill'>Xem thông tin chi tiết Khoa {department_name}</a>\n\n"
            
            # Add department data - only if exists
            for key, value in department_data.items():
                if key != "full_info" and value:  # Skip full info and empty values
                    if isinstance(value, list):
                        context += f"- {key.replace('_', ' ').title()}: {', '.join(str(v) for v in value)}\n"
                    else:
                        context += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            # Add doctors in this department with links - only if exists
            department_doctors = [s for s in self.specialists.values() 
                                if s.get("department") == department_name]
            if department_doctors:
                context += "\n- Đội ngũ bác sĩ:\n"
                for doctor in department_doctors:
                    doctor_name = doctor.get("name", "")
                    if doctor_name:
                        qualifications = doctor.get("qualifications", "")
                        experience = doctor.get("experience", "")
                        
                        display_name = f"{qualifications} {doctor_name}" if qualifications else doctor_name
                        context += f"  + {display_name}\n"
                        
                        if experience:
                            context += f"    Kinh nghiệm: {experience}\n"
                        
                        doctor_id = await self.get_doctor_id_by_name(doctor_name)
                        if doctor_id:
                            context += f"    👨‍⚕️ <a href='/Patient/DetailDoctor/{doctor_id}' class='doctor-detail-link btn btn-outline-primary btn-hover-fill'>Xem chi tiết</a>\n"
                        context += "\n"
        
        # Build the complete prompt for Gemini
        full_prompt = f"""
        {system_prompt}
        
        {context}
        
        Câu hỏi của người dùng: {query}
        
        Lưu ý: 
        - Khi trả lời về bệnh, tập trung vào thông tin y khoa về bệnh trước, sau đó mới giới thiệu bác sĩ chuyên khoa
        - Khi trả lời về bác sĩ, hãy chia sẻ thông tin về kinh nghiệm, chuyên môn của bác sĩ trước
        - Khi trả lời về khoa, hãy chia sẻ thông tin về chức năng, dịch vụ của khoa trước  
        - TUYỆT ĐỐI KHÔNG nói "chưa được cung cấp" hoặc "không có thông tin chi tiết"
        - CHỈ sử dụng thông tin có sẵn trong dữ liệu được cung cấp
        - Nếu không có thông tin về experience, qualifications, specialties thì BỎ QUA không nhắc đến
        - Sử dụng format HTML đẹp cho các link với class 'btn btn-outline-primary btn-hover-fill'
        - Thêm icon 👨‍⚕️ cho link bác sĩ và 🏥 cho link khoa để dễ nhận biết
        
        Trả lời:
        """
        
        try:
            # Get response from Gemini model
            response = self.gemini_model.generate_content(full_prompt)
            generated_response = response.text
            
            return generated_response
        except Exception as e:
            logging.error(f"Error generating Gemini response: {e}")
            return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
            

    async def format_retrieval_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieval results into context for generation with doctor links"""
        if not results:
            return ""
            
        context = "BỐI CẢNH:\n\n"
        
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            original_text = metadata.get("original_text", "")
            
            # Format based on metadata type
            content = await self._format_context_entry_with_links(metadata, original_text, i)
            context += content
        
        return context

    async def _format_context_entry_with_links(self, metadata, original_text, index):
        """Format a single context entry based on its type with doctor links"""        
        if metadata.get("type") == "qa":
            if metadata.get("section") == "question":
                answer = metadata.get("answer", "")
                # Add doctor links to answer if it contains doctor names
                answer = await self.add_doctor_links_to_text(answer)
                return f"Câu hỏi: {original_text}\nTrả lời: {answer}\n\n"
            else:
                related_question = metadata.get("related_question", "")
                processed_text = await self.add_doctor_links_to_text(original_text)
                if related_question:
                    return f"Câu hỏi: {related_question}\nTrả lời: {processed_text}\n\n"
                else:
                    return f"Thông tin: {processed_text}\n\n"
        else:
            source_info = ""
            processed_text = await self.add_doctor_links_to_text(original_text)
            
            if metadata.get("type") == "disease":
                source_info = f"Thông tin về bệnh {metadata.get('disease_name', '')}"
            elif metadata.get("type") == "specialist":
                specialist_name = metadata.get('specialist_name', '')
                source_info = f"Thông tin về bác sĩ {specialist_name}"
                
                # Add specialist info before the link
                if specialist_name in self.specialists:
                    specialist_data = self.specialists[specialist_name]
                    qualifications = specialist_data.get("qualifications", "")
                    experience = specialist_data.get("experience", "")
                    
                    if qualifications:
                        source_info = f"Thông tin về {qualifications} {specialist_name}"
                    
                    # Add direct link for the specialist with better styling
                    doctor_id = await self.get_doctor_id_by_name(specialist_name)
                    if doctor_id:
                        source_info += f"\n📋 <a href='/Patient/DetailDoctor/{doctor_id}' class='doctor-detail-link' style='color: #0066cc; text-decoration: none; font-weight: 500; padding: 6px 12px; border: 1px solid #0066cc; border-radius: 4px; display: inline-block; margin-top: 4px;'>» Xem hồ sơ đầy đủ</a>"
                        
            elif metadata.get("type") == "department":
                source_info = f"Thông tin về Khoa {metadata.get('department_name', '')}"
            
            if source_info:
                return f"{source_info}: {processed_text}\n\n"
            else:
                return f"Thông tin {index}: {processed_text}\n\n"

    async def add_doctor_links_to_text(self, text: str) -> str:
        """Add doctor links to any doctor names mentioned in text"""
        if not text:
            return text
            
        processed_text = text
        
        for specialist_name in self.specialists.keys():
            if specialist_name.lower() in processed_text.lower():
                doctor_id = await self.get_doctor_id_by_name(specialist_name)
                if doctor_id:
                    # Only add link if not already present
                    if f"/Patient/DetailDoctor/{doctor_id}" not in processed_text:
                        import re
                        pattern = re.compile(re.escape(specialist_name), re.IGNORECASE)
                        
                        def replace_with_link(match):
                            matched_name = match.group(0)
                            # Get specialist info
                            specialist_info = self.specialists.get(specialist_name, {})
                            qualifications = specialist_info.get("qualifications", "")
                            experience = specialist_info.get("experience", "")
                            
                            result = matched_name
                            if qualifications and qualifications not in matched_name:
                                result = f"{qualifications} {matched_name}"
                            
                            # Only add experience if it exists
                            if experience:
                                result += f" (Kinh nghiệm: {experience})"
                            
                            result += f" - 📋 <a href='/Patient/DetailDoctor/{doctor_id}' class='doctor-detail-link' style='color: #0066cc; text-decoration: none; font-weight: 500;'>Xem chi tiết</a>"
                            
                            return result
                        
                        processed_text = pattern.sub(replace_with_link, processed_text, count=1)
        
        return processed_text
    
    async def generate_fallback_response(self, query: str) -> str:
        """Generate fallback response when no relevant information is found"""
        greeting_patterns = [
            r'(xin\s+ch[àa]o|hello|hi|h[êe]y|ch[àa]o\s+b[aạ]n)',
            r'(tên\s+g[ìi]|tên\s+l[àa]|tên\s+của\s+bạn)',
            r'(bạn\s+l[àa](\s+ai)?|bạn\s+l[àa]m\s+g[ìi]|chức\s+năng)'
        ]
        
        for pattern in greeting_patterns:
            if re.search(pattern, query.lower()):
                return """
                Xin chào! Tôi là trợ lý y tế ảo của bệnh viện, chuyên cung cấp thông tin về các bệnh lý, 
                chuyên khoa và bác sĩ tại bệnh viện. Tôi có thể giúp bạn tìm hiểu về các triệu chứng, 
                phương pháp điều trị, bác sĩ chuyên khoa phù hợp và thông tin về các khoa phòng. 
                Bạn cần tư vấn về vấn đề gì?
                """
        
        if not await self.is_relevant_query(query):
            return """
            Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi chuyên môn y tế của tôi. 
            Tôi chỉ có thể cung cấp thông tin liên quan đến các vấn đề sức khỏe, bệnh lý, 
            chuyên khoa và đội ngũ y bác sĩ của bệnh viện. 
            Vui lòng đặt câu hỏi liên quan đến lĩnh vực y tế để tôi có thể hỗ trợ bạn tốt hơn.
            """
        
        return """
        Xin lỗi, tôi chưa có đủ thông tin để trả lời câu hỏi của bạn một cách chính xác. 
        Để nhận được tư vấn cụ thể, bạn có thể:
        1. Thử diễn đạt câu hỏi theo cách khác với nhiều chi tiết hơn
        2. Liên hệ trực tiếp với bộ phận tư vấn của bệnh viện qua số điện thoại hotline
        3. Đặt lịch hẹn với bác sĩ chuyên khoa phù hợp để được thăm khám trực tiếp
        
        Bạn có thể cho tôi biết thêm về vấn đề bạn đang gặp phải không?
        """
    
    async def answer(self, query: str) -> str:
        """Process query and generate answer with enhanced symptom detection"""
        # Continue with existing flow if not a symptom query or no matches found
        # Check for direct pattern handling
        direct_answer = await self.handle_specialized_queries(query)
        if direct_answer:
            return direct_answer
        
        # Determine query type for specialized handling
        query_type = await self.determine_query_type(query)
        
        # Get retrieved results based on query type
        results = await self.hybrid_search(query, filter_type=query_type if query_type != "general" else None)
        
        # If no results found, try without filtering
        if not results and query_type != "general":
            results = await self.hybrid_search(query)
        
        # Merge adjacent chunks for better context
        results = await self.auto_merge_chunks(results)
        
        # If still no relevant results, return fallback response
        if not results:
            return await self.generate_fallback_response(query)
        
        # Generate response using retrieved context
        return await self.generate_response(query, results, query_type)

    async def generate_symptom_response(self, query: str, predicted_diseases: List[Dict[str, Any]]) -> str:
        """Generate definitive disease prediction response based on symptoms"""
        if not predicted_diseases:
            return """
            Dựa trên các triệu chứng bạn mô tả, tôi không tìm thấy bệnh lý cụ thể nào phù hợp trong cơ sở dữ liệu. 
            Điều này có thể do:
            - Triệu chứng chưa đủ chi tiết hoặc cụ thể
            - Có thể là bệnh lý hiếm gặp không có trong dữ liệu
            - Cần thêm thông tin về thời gian, mức độ nghiêm trọng
            
            🏥 **Khuyến nghị**: Bạn nên thăm khám trực tiếp với bác sĩ đa khoa để được chẩn đoán chính xác.
            """
        
        # Start with confident disease prediction
        top_disease = predicted_diseases[0]
        response = ""
        response += f"**Bạn có khả năng cao nhất đang mắc: {top_disease['disease_name']}**\n\n"
        
        # Show matched symptoms first
        matched_symptoms = top_disease['matched_symptoms']
        if matched_symptoms:
            response += f"✅ **Triệu chứng bạn có phù hợp với bệnh này**: {', '.join(matched_symptoms)}\n\n"
        
        # Add comprehensive symptom list for the top disease
        disease_data = top_disease['disease_data']        
        # Add disease information
        disease_description = ""
        qa_pairs = disease_data.get("qa_pairs", [])
        for qa_pair in qa_pairs:
            question = qa_pair.get("question", "").lower()
            if any(keyword in question for keyword in ["mô tả", "định nghĩa", "triệu chứng", "biểu hiện"]):
                disease_description = qa_pair.get("answer", "")
                break
        
        if not disease_description and "description" in disease_data:
            disease_description = disease_data["description"]
        
        if disease_description:
            response += f"📖 **Thông tin về bệnh**:\n{disease_description}\n\n"
        
        # Show other possible diseases
        if len(predicted_diseases) > 1:
            response += "🔄 **Các khả năng khác cần xem xét**:\n"
            for i, disease in enumerate(predicted_diseases[1:4], 2):  # Show next 3
                response += f"{i}. **{disease['disease_name']}** "
                if disease['matched_symptoms']:
                    response += f"(Triệu chứng trùng: {', '.join(disease['matched_symptoms'][:3])})\n"
                else:
                    response += "\n"
            response += "\n"
        
        # Get treatment recommendations
        recommended_departments = await self._get_departments_for_diseases(predicted_diseases[:3])
        recommended_doctors = await self._get_doctors_for_diseases(predicted_diseases[:3])
        
        # Department recommendations
        if recommended_departments:
            response += "🏥 **KHOA CHUYÊN MÔN ĐIỀU TRỊ**:\n\n"
            
            # Prioritize departments by relevance
            dept_list = list(recommended_departments.items())
            dept_list.sort(key=lambda x: len(x[1]['diseases_treated']), reverse=True)
            
            for dept_name, dept_info in dept_list[:3]:  # Top 3 departments
                response += f"**🔹 Khoa {dept_name}**\n"
                response += f"📌 Chuyên điều trị: {', '.join(dept_info['diseases_treated'])}\n"
                
                if dept_info['department_id']:
                    response += f"🔗 <a href='/Patient/DetailSpecialities/{dept_info['department_id']}' class='btn btn-outline-primary btn-hover-fill'>Xem thông tin khoa</a>\n"
                response += "\n"
        
        # Doctor recommendations with priority
        if recommended_doctors:
            response += "👨‍⚕️ **BÁC SĨ ĐƯỢC KHUYẾN NGHỊ**:\n\n"
            
            # Group and prioritize doctors
            priority_doctors = []
            other_doctors = []
            
            for doctor in recommended_doctors[:8]:  # Limit to 8 doctors
                if top_disease['disease_name'] in doctor['diseases_can_treat']:
                    priority_doctors.append(doctor)
                else:
                    other_doctors.append(doctor)
            
            # Show priority doctors first
            if priority_doctors:
                response += "⭐ **Bác sĩ chuyên điều trị bệnh này**:\n"
                for doctor in priority_doctors[:3]:  # Top 3 priority
                    display_name = f"{doctor['qualifications']} {doctor['name']}" if doctor['qualifications'] else doctor['name']
                    response += f"**• {display_name}**"
                    
                    if doctor['department']:
                        response += f" - Khoa {doctor['department']}"
                    response += "\n"
                    
                    if doctor['experience']:
                        response += f"  💼 Kinh nghiệm: {doctor['experience']}\n"
                    
                    if doctor['specialties']:
                        specialties = doctor['specialties'] if isinstance(doctor['specialties'], list) else [doctor['specialties']]
                        response += f"  🎯 Chuyên môn: {', '.join(specialties[:2])}\n"
                    
                    if doctor['doctor_id']:
                        response += f"  📞 <a href='/Patient/DetailDoctor/{doctor['doctor_id']}' class='btn btn-success'>🚀 Đặt lịch ngay</a>\n"
                    response += "\n"
            
            # Show other relevant doctors
            if other_doctors and len(priority_doctors) < 3:
                remaining_slots = 3 - len(priority_doctors)
                response += "\n📋 **Bác sĩ có thể hỗ trợ**:\n"
                for doctor in other_doctors[:remaining_slots]:
                    display_name = f"{doctor['qualifications']} {doctor['name']}" if doctor['qualifications'] else doctor['name']
                    response += f"• {display_name}"
                    if doctor['department']:
                        response += f" - Khoa {doctor['department']}"
                    
                    if doctor['doctor_id']:
                        response += f" - <a href='/Patient/DetailDoctor/{doctor['doctor_id']}' class='btn btn-outline-primary'>Xem chi tiết</a>"
                    response += "\n"
        
        # Important notes
        response += "\n⚠️ **LƯU Ý QUAN TRỌNG**:\n"
        response += "• Đây là dự đoán dựa trên triệu chứng, không thay thế chẩn đoán y khoa\n"
        response += "• Việc chẩn đoán chính xác cần khám lâm sàng và xét nghiệm\n"
        response += "• Nếu triệu chứng nặng hoặc có biến chứng, đến bệnh viện ngay\n"
        response += "• Tuân thủ hướng dẫn của bác sĩ khi đã có kết quả khám\n"
        
        return response
    
    async def _get_all_symptoms_for_disease(self, disease_data: Dict[str, Any]) -> List[str]:
        """Extract all symptoms for a disease from various data sources"""
        symptoms = []
        
        # Get symptoms from symptoms field
        if 'symptoms' in disease_data:
            disease_symptoms = disease_data['symptoms']
            if isinstance(disease_symptoms, list):
                symptoms.extend(disease_symptoms)
            elif isinstance(disease_symptoms, str):
                # Split by common delimiters
                symptoms.extend([s.strip() for s in disease_symptoms.split(',')])
        
        # Get symptoms from QA pairs
        qa_pairs = disease_data.get("qa_pairs", [])
        for qa_pair in qa_pairs:
            question = qa_pair.get("question", "").lower()
            answer = qa_pair.get("answer", "")
            
            # Look for symptom-related questions
            if any(keyword in question for keyword in ["triệu chứng", "biểu hiện", "dấu hiệu", "symptoms"]):
                # Extract symptoms from answer
                symptom_matches = await self.extract_symptoms_from_text(answer)
                symptoms.extend(symptom_matches)
        
        # Get symptoms from description
        if 'description' in disease_data:
            description_symptoms = await self.extract_symptoms_from_text(disease_data['description'])
            symptoms.extend(description_symptoms)
        
        # Clean and deduplicate symptoms
        cleaned_symptoms = []
        for symptom in symptoms:
            cleaned = symptom.strip().strip('•-*').strip()
            if cleaned and len(cleaned) > 2:  # Filter out very short strings
                cleaned_symptoms.append(cleaned)
        
        # Remove duplicates while preserving order
        unique_symptoms = []
        seen = set()
        for symptom in cleaned_symptoms:
            symptom_lower = symptom.lower()
            if symptom_lower not in seen:
                seen.add(symptom_lower)
                unique_symptoms.append(symptom)
        
        return unique_symptoms
    
    async def _extract_by_patterns(self, query: str) -> List[str]:
        """
        Trích xuất triệu chứng bằng các pattern regex
        """
        symptoms = []
        
        # Pattern 1: "đau [vị trí]" - đau + vị trí cơ thể
        pain_pattern = r'đau\s+([\w\s]+?)(?=\s*(?:và|,|;|\.|$))'
        pain_matches = re.findall(pain_pattern, query)
        for match in pain_matches:
            clean_match = await self._clean_symptom_text(match.strip())
            if clean_match:
                symptoms.append(f"đau {clean_match}")
        
        # Pattern 2: "bị [triệu chứng]"
        suffer_pattern = r'bị\s+([\w\s]+?)(?=\s*(?:và|,|;|\.|$))'
        suffer_matches = re.findall(suffer_pattern, query)
        for match in suffer_matches:
            clean_match = await self._clean_symptom_text(match.strip())
            if clean_match:
                symptoms.append(clean_match)
        
        # Pattern 3: "cảm thấy [triệu chứng]"
        feel_pattern = r'cảm\s+thấy\s+([\w\s]+?)(?=\s*(?:và|,|;|\.|$))'
        feel_matches = re.findall(feel_pattern, query)
        for match in feel_matches:
            clean_match = await self._clean_symptom_text(match.strip())
            if clean_match:
                symptoms.append(clean_match)
        
        # Pattern 4: "có [triệu chứng]"
        have_pattern = r'có\s+([\w\s]+?)(?=\s*(?:và|,|;|\.|$))'
        have_matches = re.findall(have_pattern, query)
        for match in have_matches:
            clean_match = await self._clean_symptom_text(match.strip())
            if clean_match and len(clean_match) > 2:  # Loại bỏ từ quá ngắn
                symptoms.append(clean_match)
        
        return symptoms
    
    async def _clean_symptom_text(self, text: str) -> str:
        """
        Làm sạch text triệu chứng
        """
        # Loại bỏ các từ stop words
        words = text.split()
        cleaned_words = [word for word in words if word not in self.stop_words]
        
        # Loại bỏ các ký tự đặc biệt
        cleaned_text = ' '.join(cleaned_words)
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
        
        return cleaned_text.strip()
    
    async def _is_valid_symptom(self, text: str) -> bool:
        """
        Kiểm tra xem text có phải là triệu chứng hợp lệ không
        """
        if len(text) < 3:  # Quá ngắn
            return False
            
        # Kiểm tra có chứa từ khóa y tế
        medical_keywords = [
            'đau', 'nhức', 'sốt', 'ho', 'khó', 'nôn', 'chảy', 'mỏi', 
            'mệt', 'chóng', 'ngứa', 'sưng', 'ban', 'viêm', 'rát'
        ]
        
        for keyword in medical_keywords:
            if keyword in text:
                return True
        
        # Kiểm tra có phải là triệu chứng phổ biến
        for symptom in self.common_symptoms:
            if symptom in text or text in symptom:
                return True
        
        return False

    def _is_valid_symptom(self, text: str) -> bool:
        """
        Kiểm tra xem text có phải là triệu chứng hợp lệ không
        """
        if len(text) < 3:  # Quá ngắn
            return False
            
        # Kiểm tra có chứa từ khóa y tế
        medical_keywords = [
            'đau', 'nhức', 'sốt', 'ho', 'khó', 'nôn', 'chảy', 'mỏi', 
            'mệt', 'chóng', 'ngứa', 'sưng', 'ban', 'viêm', 'rát'
        ]
        
        for keyword in medical_keywords:
            if keyword in text:
                return True
        
        # Kiểm tra có phải là triệu chứng phổ biến
        for symptom in self.common_symptoms:
            if symptom in text or text in symptom:
                return True
        
        return False
    
    async def _extract_by_grammar(self, query: str) -> List[str]:
        """
        Trích xuất triệu chứng dựa trên cấu trúc ngữ pháp
        """
        symptoms = []
        
        # Tách câu thành các phần bằng dấu phẩy, "và"
        parts = re.split(r'\s*(?:và|,|;)\s*', query)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Loại bỏ các từ không cần thiết ở đầu
            part = re.sub(r'^(?:tôi|mình|em|anh|chị|bác)\s+', '', part)
            part = re.sub(r'^(?:bị|có|cảm|thấy|giác)\s+', '', part)
            
            # Nếu phần còn lại là triệu chứng hợp lệ
            if self._is_valid_symptom(part):
                symptoms.append(part)
        
        return symptoms

    async def extract_symptoms_from_text(self, query: str) -> List[str]:
        """
        Trích xuất triệu chứng từ văn bản một cách thông minh
        """
        query_lower = query.lower()
        extracted_symptoms = []
        
        # Bước 1: Tìm triệu chứng trực tiếp từ danh sách
        for symptom in self.common_symptoms:
            if symptom in query_lower:
                if symptom not in extracted_symptoms:
                    extracted_symptoms.append(symptom)
        
        # Bước 2: Trích xuất triệu chứng bằng pattern matching
        additional_symptoms = await self._extract_by_patterns(query_lower)
        for symptom in additional_symptoms:
            if symptom not in extracted_symptoms:
                extracted_symptoms.append(symptom)
        
        # Bước 3: Trích xuất từ cấu trúc ngữ pháp
        grammar_symptoms = await self._extract_by_grammar(query_lower)
        for symptom in grammar_symptoms:
            if symptom not in extracted_symptoms:
                extracted_symptoms.append(symptom)
        
        return extracted_symptoms
    
    async def handle_symptom_query(self, query: str) -> Optional[str]:
        """
        Xử lý câu hỏi về triệu chứng - ưu tiên trả lời thông tin thay vì dự đoán
        """
        query_lower = query.lower()
        
        # Danh sách từ khóa cho câu hỏi về thông tin bệnh
        info_keywords = [
            'triệu chứng của', 'triệu chứng bệnh', 'biểu hiện của', 'dấu hiệu of',
            'bệnh gì có triệu chứng', 'có triệu chứng gì', 'có biểu hiện gì', 
            'có dấu hiệu gì', 'làm sao biết', 'nhận biết', 'phân biệt'
        ]
        
        # Danh sách từ khóa cho việc mô tả triệu chứng cá nhân
        personal_keywords = [
            'tôi bị', 'mình bị', 'em bị', 'con bị', 'bé bị',
            'tôi có', 'mình có', 'em có', 'tôi cảm thấy', 'mình cảm thấy',
            'tôi đang', 'mình đang', 'này là bệnh gì', 'có phải bị'
        ]
        
        # Kiểm tra xem đây có phải câu hỏi về thông tin bệnh không
        is_info_question = any(keyword in query_lower for keyword in info_keywords)
        
        # Kiểm tra xem có phải mô tả triệu chứng cá nhân không  
        is_personal_symptoms = any(keyword in query_lower for keyword in personal_keywords)
        
        # Ưu tiên xử lý câu hỏi thông tin trước
        if is_info_question and not is_personal_symptoms:
            # Đây là câu hỏi về thông tin bệnh - không dự đoán
            return None  # Để cho phần xử lý thông tin bệnh khác handle
        
        # Chỉ khi rõ ràng là mô tả triệu chứng cá nhân mới dự đoán
        elif is_personal_symptoms:
            predicted_diseases = await self.predict_disease_from_symptoms(query)
            if predicted_diseases:
                return await self.generate_symptom_response(query, predicted_diseases)
            else:
                return """
                Tôi nhận thấy bạn đang mô tả một số triệu chứng, nhưng cần thêm thông tin cụ thể để đưa ra dự đoán chính xác hơn.
                
                🔍 **Để hỗ trợ tốt hơn, hãy cho biết**:
                - Các triệu chứng cụ thể (sốt, ho, đau đầu, buồn nôn...)
                - Thời gian xuất hiện (bao lâu rồi)
                - Mức độ nghiêm trọng (nhẹ, trung bình, nặng)
                - Có yếu tố khởi phát nào không
                
                🏥 **Hoặc bạn có thể**:
                - Đặt lịch khám với bác sĩ đa khoa
                - Gọi tổng đài tư vấn sức khỏe
                - Đến khoa cấp cứu nếu triệu chứng nghiêm trọng
                """
        
        # Trường hợp không rõ ràng - không xử lý
        return None
    
    async def initialize(self) -> bool:
        """Initialize the chatbot by loading data and creating indexes"""

        if not hasattr(self, 'history_manager'):
            self.history_manager = ChatHistoryManager(self.sql_config)
            print("History manager initialized:", self.history_manager)
        # Check if cache exists
        cache_file = os.path.join(self.config["cache_dir"], "medical_chatbot_cache_final.pkl")
        
        if os.path.exists(cache_file):
            try:
                logging.info("Loading from cache...")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Load cached data
                self.corpus_texts = cache_data.get("corpus_texts", [])
                self.corpus_embeddings = cache_data.get("corpus_embeddings", None)
                self.corpus_metadata = cache_data.get("corpus_metadata", [])
                self.bm25_index = cache_data.get("bm25_index", None)
                self.diseases = cache_data.get("diseases", {})
                self.departments = cache_data.get("departments", {})
                self.specialists = cache_data.get("specialists", {})
                
                # Recreate Qdrant index
                await self._recreate_qdrant_index()
                
                logging.info(f"Successfully loaded cached data with {len(self.corpus_texts)} chunks")
                return True
                
            except Exception as e:
                logging.error(f"Error loading from cache: {e}")
        
        # Normal initialization
        success = await self.load_medical_data()
        if success:
            await self.create_indexes()
            
            # Save to cache
            try:
                await self._save_cache(cache_file)
            except Exception as e:
                logging.error(f"Error saving cache: {e}")
                
            return True
        else:
            logging.error("Failed to load medical data")
            return False
    
    async def _save_cache(self, cache_file):
        cache_data = {
            "corpus_texts": self.corpus_texts,
            "corpus_embeddings": self.corpus_embeddings,
            "corpus_metadata": self.corpus_metadata,
            "bm25_index": self.bm25_index,
            "diseases": self.diseases,
            "departments": self.departments,
            "specialists": self.specialists
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
            
        logging.info(f"Saved cache to {cache_file}")
    
    async def _recreate_qdrant_index(self):
        if self.corpus_embeddings is not None:
            vector_size = self.corpus_embeddings.shape[1]
            
            try:
                self.qdrant_client.delete_collection(self.collection_name)
            except:
                pass
                
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            
            # Load data into Qdrant
            batch_size = 100
            points = []
            for i, (embedding, metadata) in enumerate(zip(self.corpus_embeddings, self.corpus_metadata)):
                points.append(PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload=metadata
                ))
                
                # Insert in batches
                if len(points) >= batch_size or i == len(self.corpus_embeddings) - 1:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    points = []
        
    async def evaluate_response(self, query: str, answer: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        evaluation = {
            "query": query,
            "answer": answer,
            "context_precision": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "extracted_context": []
        }

        # Trích xuất ngữ cảnh gốc
        for ctx in retrieved_context:
            metadata = ctx.get("metadata", {})
            original_text = metadata.get("original_text", ctx.get("text", ""))
            evaluation["extracted_context"].append({
                "text": original_text,
                "metadata": {
                    "type": metadata.get("type", ""),
                    "disease_name": metadata.get("disease_name", ""),
                    "specialist_name": metadata.get("specialist_name", ""),
                    "department_name": metadata.get("department_name", ""),
                    "score": ctx.get("score", 0.0)
                }
            })

        # 1. Context Precision: Đánh giá tỷ lệ các ngữ cảnh được truy xuất là phù hợp với query
        relevant_context_count = 0
        total_context_count = len(retrieved_context) if retrieved_context else 1  # Tránh chia cho 0
        
        query_tokens = set((await self.preprocess_text(query)).split())
        for ctx in retrieved_context:
            ctx_text = ctx.get("text", "")
            ctx_tokens = set(ctx_text.split())
            # Đếm số token chung giữa query và context
            common_tokens = len(query_tokens.intersection(ctx_tokens))
            # Context được coi là phù hợp nếu có ít nhất 20% token chung hoặc score cao
            if common_tokens / len(query_tokens) >= 0.2 or ctx.get("score", 0.0) > 0.8:
                relevant_context_count += 1
        
        evaluation["context_precision"] = relevant_context_count / total_context_count

        # 2. Faithfulness: Đánh giá mức độ câu trả lời dựa trên ngữ cảnh
        answer_embedding = self.embedding_model.encode([answer])[0]
        context_texts = [ctx.get("metadata", {}).get("original_text", ctx.get("text", "")) for ctx in retrieved_context]
        context_embeddings = self.embedding_model.encode(context_texts) if context_texts else np.zeros((1, answer_embedding.shape[0]))
        
        similarities = []
        for ctx_emb in context_embeddings:
            sim = np.dot(answer_embedding, ctx_emb) / (np.linalg.norm(answer_embedding) * np.linalg.norm(ctx_emb))
            similarities.append(sim)
        
        evaluation["faithfulness"] = float(np.mean(similarities)) if similarities else 0.0

        # 3. Answer Relevancy: Đánh giá mức độ câu trả lời liên quan đến câu hỏi
        relevancy_score = self.cross_encoder.predict([(query, answer)])[0]
        evaluation["answer_relevancy"] = float(relevancy_score)

        # Chuẩn hóa các điểm số về [0, 1]
        for metric in ["context_precision", "faithfulness", "answer_relevancy"]:
            evaluation[metric] = min(max(evaluation[metric], 0.0), 1.0)

        return evaluation

    async def answer_with_evaluation(self, query: str) -> Dict[str, Any]:
        """
        Trả lời câu hỏi và đánh giá câu trả lời theo các tiêu chí.
        
        Args:
            query (str): Câu hỏi của người dùng.
        
        Returns:
            Dict[str, Any]: Câu trả lời và kết quả đánh giá.
        """
        start_time = time.time()
        
        # Kiểm tra xử lý các câu hỏi đặc biệt
        direct_answer = await self.handle_specialized_queries(query)
        if direct_answer:
            return {
                "response": direct_answer,
                "evaluation": {
                    "query": query,
                    "answer": direct_answer,
                    "context_precision": 1.0,  # Giả định direct answer là chính xác
                    "faithfulness": 1.0,
                    "answer_relevancy": 1.0,
                    "extracted_context": []
                },
                "processing_time": time.time() - start_time
            }
        
        # Xác định loại câu hỏi
        query_type = await self.determine_query_type(query)
        
        # Truy xuất ngữ cảnh
        results = await self.hybrid_search(query, filter_type=query_type if query_type != "general" else None)
        if not results and query_type != "general":
            results = await self.hybrid_search(query)
        
        # Gộp các đoạn ngữ cảnh liền kề
        results = await self.auto_merge_chunks(results)
        
        # Tạo câu trả lời
        if not results:
            answer = await self.generate_fallback_response(query)
        else:
            answer = await self.generate_response(query, results, query_type)
        
        # Đánh giá câu trả lời
        evaluation = await self.evaluate_response(query, answer, results)
        
        return {
            "response": answer,
            "evaluation": evaluation,
            "processing_time": time.time() - start_time
        }

    async def chat(self):
        """Interactive chat interface for the chatbot with evaluation"""
        if not await self.initialize():
            print("Failed to initialize chatbot. Exiting.")
            return
            
        print("=" * 80)
        print("Trợ lý Y tế Ảo - Bệnh viện Chuyên khoa")
        print("Gõ 'exit', 'quit' hoặc 'bye' để thoát.")
        print("=" * 80)
        
        while True:
            user_input = input("\nBạn: ")
            
            if user_input.lower() in ["exit", "quit", "bye", "thoát"]:
                print("Cảm ơn bạn đã sử dụng dịch vụ. Chúc bạn sức khỏe!")
                break
                
            # Gọi hàm answer_with_evaluation
            result = await self.answer_with_evaluation(user_input)
            
            print(f"\nTrợ lý Y tế: {result['response']}")
            print(f"\n[Thời gian phản hồi: {result['processing_time']:.2f}s]")
            
            # In kết quả đánh giá
            print("\nĐánh giá câu trả lời:")
            print(f"- Context Precision: {result['evaluation']['context_precision']:.2f}")
            print(f"- Faithfulness: {result['evaluation']['faithfulness']:.2f}")
            print(f"- Answer Relevancy: {result['evaluation']['answer_relevancy']:.2f}")
            print("\nNgữ cảnh được trích xuất:")
            for ctx in result['evaluation']['extracted_context']:
                print(f"- {ctx['text'][:100]}... (Type: {ctx['metadata']['type']}, Score: {ctx['metadata']['score']:.2f})")
if __name__ == "__main__":
    chatbot = MedicalSpecialistRAGChatbot()
    asyncio.run(chatbot.chat())
