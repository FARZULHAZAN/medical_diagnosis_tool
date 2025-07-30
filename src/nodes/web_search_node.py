import requests
import json
import logging
from typing import List
from src.state.Agentstate import AgentState
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalWebSearchAgent:
    def __init__(self):
        """Initialize the medical web search agent"""
        # Hardcoded API keys
        self.serper_api_key = os.getenv('serper_api_key')
        self.serpapi_key = os.getenv('serpapi_key ')

    def _search_web(self, query: str, max_results: int = 5) -> str:
        """
        Dynamic web search using multiple search engines with fallback
        Returns combined search results as text
        """
        # Try Serper first
        try:
            logger.info(f"Trying Serper search for: {query}")
            url = "https://google.serper.dev/search"
            payload = json.dumps({
                "q": query,
                "num": max_results,
                "gl": "us",
                "hl": "en"
            })
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('organic', []):
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                results.append(f"{title}. {snippet}")
            
            if results:
                logger.info(f"Serper search successful - {len(results)} results")
                return " ".join(results)
                
        except Exception as e:
            logger.warning(f"Serper search failed: {e}")

        # Try SerpAPI second
        try:
            logger.info(f"Trying SerpAPI search for: {query}")
            params = {
                'q': query,
                'api_key': self.serpapi_key,
                'engine': 'google',
                'num': max_results,
                'gl': 'us',
                'hl': 'en'
            }
            
            response = requests.get('https://serpapi.com/search', params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('organic_results', []):
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                results.append(f"{title}. {snippet}")
            
            if results:
                logger.info(f"SerpAPI search successful - {len(results)} results")
                return " ".join(results)
                
        except Exception as e:
            logger.warning(f"SerpAPI search failed: {e}")

        # Try DuckDuckGo last
        try:
            logger.info(f"Trying DuckDuckGo search for: {query}")
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                
                for result in search_results:
                    title = result.get('title', '')
                    body = result.get('body', '')
                    results.append(f"{title}. {body}")
            
            if results:
                logger.info(f"DuckDuckGo search successful - {len(results)} results")
                return " ".join(results)
                
        except ImportError:
            logger.error("DuckDuckGo search requires: pip install duckduckgo-search")
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")

        # All searches failed
        logger.error("All search methods failed")
        return ""

    def search_disease(self, state: AgentState) -> AgentState:
        """
        Search for disease/condition based on extracted symptoms
        """
        symptoms = state.get("extracted_symptoms", [])
        
        if not symptoms:
            logger.warning("No symptoms found in state")
            state["retrieved_disease"] = {
                "name": "No symptoms provided",
                "symptoms": [],
                "search_results": ""
            }
            return state
        
        try:
            # Build dynamic search query from symptoms
            symptoms_text = " ".join([str(symptom).strip() for symptom in symptoms if str(symptom).strip()])
            search_query = f"medical condition disease diagnosis symptoms {symptoms_text}"
            
            logger.info(f"Searching for disease with query: {search_query}")
            
            # Get search results
            search_results = self._search_web(search_query, max_results=5)
            
            if search_results:
                # Extract disease name dynamically from search results
                disease_name = self._extract_disease_name_from_text(search_results, symptoms)
                
                state["retrieved_disease"] = {
                    "name": disease_name,
                    "symptoms": symptoms,
                    "search_results": search_results[:1000]  # Limit for storage
                }
                logger.info(f"Disease search completed: {disease_name}")
            else:
                state["retrieved_disease"] = {
                    "name": "Unable to determine from search",
                    "symptoms": symptoms,
                    "search_results": ""
                }
                
        except Exception as e:
            logger.error(f"Disease search error: {e}")
            state["retrieved_disease"] = {
                "name": "Search error occurred",
                "symptoms": symptoms,
                "search_results": str(e)
            }
        
        return state

    def search_medicines(self, state: AgentState) -> AgentState:
        """
        Search for medicines based on identified disease or symptoms
        """
        disease_info = state.get("retrieved_disease", {})
        disease_name = disease_info.get("name", "")
        symptoms = disease_info.get("symptoms", [])
        
        if not disease_name and not symptoms:
            logger.warning("No disease or symptoms found for medicine search")
            state["medicines"] = []
            state["medicine_search_results"] = ""
            return state
        
        try:
            # Build dynamic search query for medicines
            if disease_name and disease_name not in ["No symptoms provided", "Unable to determine from search", "Search error occurred"]:
                search_query = f"medications drugs medicine treatment for {disease_name}"
            else:
                symptoms_text = " ".join([str(symptom).strip() for symptom in symptoms if str(symptom).strip()])
                search_query = f"medications drugs medicine for symptoms {symptoms_text}"
            
            logger.info(f"Searching for medicines with query: {search_query}")
            
            # Get search results
            search_results = self._search_web(search_query, max_results=5)
            
            if search_results:
                # Extract medicines dynamically from search results
                medicines = self._extract_medicines_from_text(search_results)
                
                state["medicines"] = medicines
                state["medicine_search_results"] = search_results[:1000]  # Limit for storage
                
                # Create final response
                response = f"**Disease/Condition:** {disease_name}\n\n"
                response += "**Related Medications Found:**\n"
                
                for i, medicine in enumerate(medicines, 1):
                    response += f"{i}. {medicine}\n"
                
                if not medicines:
                    response += "No specific medications identified from search results.\n"
                
                response += "\n**Note:** Always consult a healthcare provider before taking any medication."
                
                state["final_response"] = response
                logger.info(f"Medicine search completed: {len(medicines)} medicines found")
            else:
                state["medicines"] = []
                state["medicine_search_results"] = ""
                state["final_response"] = f"**Disease/Condition:** {disease_name}\n\n**Medicine Search:** No results found."
                
        except Exception as e:
            logger.error(f"Medicine search error: {e}")
            state["medicines"] = []
            state["medicine_search_results"] = str(e)
            state["final_response"] = f"**Disease/Condition:** {disease_name}\n\n**Medicine Search Error:** {str(e)}"
        
        return state

    def _extract_disease_name_from_text(self, text: str, symptoms: List[str]) -> str:
        """
        Dynamically extract disease name from search results text
        """
        if not text:
            return "No search results available"
        
        text_lower = text.lower()
        
        # Look for common medical condition patterns
        patterns_to_find = [
            # Direct mentions
            "condition", "disease", "syndrome", "disorder", "infection",
            # Specific conditions that might appear
            "flu", "influenza", "cold", "fever", "headache", "migraine",
            "pneumonia", "bronchitis", "gastroenteritis", "allergic reaction"
        ]
        
        # Find sentences that mention medical conditions
        sentences = text.split('.')
        best_sentence = ""
        
        for sentence in sentences[:10]:  # Check first 10 sentences
            sentence_lower = sentence.lower()
            
            # Count medical keywords in this sentence
            medical_score = 0
            for pattern in patterns_to_find:
                if pattern in sentence_lower:
                    medical_score += 1
            
            # Check if symptoms are mentioned too
            symptom_score = 0
            for symptom in symptoms:
                if str(symptom).lower() in sentence_lower:
                    symptom_score += 1
            
            # If this sentence has good medical + symptom relevance
            if (medical_score > 0 and symptom_score > 0) or medical_score > 1:
                best_sentence = sentence.strip()
                break
        
        if best_sentence:
            # Try to extract the actual condition name
            words = best_sentence.split()
            
            # Look for patterns like "symptoms of [condition]" or "condition called [name]"
            for i, word in enumerate(words):
                if word.lower() in ['of', 'called', 'is', 'be'] and i + 1 < len(words):
                    # Take next 1-3 words as potential condition name
                    potential_condition = " ".join(words[i+1:i+4]).strip(".,;:()")
                    if len(potential_condition) > 2:
                        return potential_condition.title()
            
            # If no clear pattern, return first few words of the best sentence
            return best_sentence[:50] + "..." if len(best_sentence) > 50 else best_sentence
        
        return "Medical condition requiring evaluation"

    def _extract_medicines_from_text(self, text: str) -> List[str]:
        """
        Accurately extract medicine names from search results text
        """
        if not text:
            return []
        
        text_lower = text.lower()
        medicines_found = set()
        
        # Known medicine name patterns (more specific)
        known_medicines = {
            # Pain/Fever medicines
            'acetaminophen', 'paracetamol', 'tylenol', 'ibuprofen', 'advil', 'motrin',
            'aspirin', 'naproxen', 'aleve', 'diclofenac', 'celecoxib',
            
            # Antibiotics
            'amoxicillin', 'penicillin', 'azithromycin', 'ciprofloxacin', 'doxycycline',
            'cephalexin', 'clarithromycin', 'metronidazole', 'trimethoprim', 'erythromycin',
            
            # Antivirals
            'acyclovir', 'valacyclovir', 'oseltamivir', 'tamiflu', 'zanamivir',
            
            # Allergy medicines
            'benadryl', 'diphenhydramine', 'loratadine', 'claritin', 'cetirizine',
            'zyrtec', 'fexofenadine', 'allegra', 'chlorpheniramine',
            
            # Cough/Cold
            'dextromethorphan', 'guaifenesin', 'pseudoephedrine', 'phenylephrine',
            'codeine', 'robitussin', 'mucinex',
            
            # Stomach medicines
            'omeprazole', 'prilosec', 'ranitidine', 'famotidine', 'pepcid',
            'lansoprazole', 'pantoprazole', 'esomeprazole', 'nexium',
            
            # Other common medicines
            'prednisone', 'hydrocortisone', 'prednisolone', 'methylprednisolone'
        }
        
        # Words to definitely exclude (not medicines)
        exclude_words = {
            'diagnosis', 'reference', 'medscape', 'list', 'missing', 'medications',
            'meningitis', 'brand', 'treatment', 'medicine', 'drug', 'drugs',
            'tablet', 'pill', 'capsule', 'syrup', 'cream', 'ointment',
            'the', 'this', 'that', 'with', 'for', 'and', 'but', 'when', 'where',
            'what', 'how', 'pain', 'relief', 'care', 'health', 'medical', 'doctor',
            'patient', 'hospital', 'clinic', 'pharmacy', 'prescription', 'over',
            'counter', 'dose', 'dosage', 'side', 'effects', 'symptoms', 'condition',
            'disease', 'infection', 'bacteria', 'virus', 'fever', 'headache',
            'nausea', 'vomiting', 'diarrhea', 'cough', 'cold', 'flu', 'allergy',
            'allergic', 'reaction', 'inflammatory', 'anti', 'inflammation'
        }
        
        # Split text into words and clean them
        words = text.replace('.', ' ').replace(',', ' ').replace(';', ' ').split()
        
        for word in words:
            # Clean the word
            clean_word = word.strip(".,;:()[]\"'!?").lower()
            
            # Skip if empty or too short
            if len(clean_word) < 4:
                continue
            
            # Check if it's a known medicine
            if clean_word in known_medicines:
                # Get the proper capitalized version
                medicines_found.add(clean_word.title())
                continue
            
            # Skip if it's in exclude list
            if clean_word in exclude_words:
                continue
            
            # Look for medicine-like patterns
            # 1. Words ending in common medicine suffixes
            medicine_suffixes = [
                'cillin',    # penicillin, amoxicillin
                'mycin',     # azithromycin, erythromycin  
                'cyclovir',  # acyclovir, valacyclovir
                'prazole',   # omeprazole, lansoprazole
                'tidine',    # ranitidine, famotidine
                'phylline',  # theophylline
                'olol',      # propranolol, metoprolol
                'pril',      # lisinopril, enalapril
                'statin',    # atorvastatin, simvastatin
                'sartan',    # losartan, valsartan
                'zole',      # metronidazole (but filter carefully)
                'pine',      # amlodipine, nifedipine
                'sone',      # prednisone, prednisolone
                'mab'        # rituximab, infliximab (antibodies)
            ]
            
            # Check if word ends with medicine suffix and isn't excluded
            for suffix in medicine_suffixes:
                if clean_word.endswith(suffix) and len(clean_word) > len(suffix) + 2:
                    # Double-check it's not in exclude list
                    if clean_word not in exclude_words:
                        medicines_found.add(clean_word.title())
                        break
        
        # Additional context-based extraction
        sentences = text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Look for specific patterns like "take [medicine]", "prescribed [medicine]"
            medicine_context_patterns = [
                'take ', 'taking ', 'use ', 'using ', 'prescribed ', 'prescribe ',
                'given ', 'administer ', 'treatment with ', 'therapy with '
            ]
            
            for pattern in medicine_context_patterns:
                if pattern in sentence_lower:
                    # Find the word(s) after the pattern
                    pattern_index = sentence_lower.find(pattern)
                    remaining_text = sentence[pattern_index + len(pattern):].strip()
                    
                    # Get the next 1-2 words as potential medicine
                    next_words = remaining_text.split()[:2]
                    for word in next_words:
                        clean_word = word.strip(".,;:()[]\"'!?").lower()
                        
                        # Check if it looks like a medicine and isn't excluded
                        if (len(clean_word) >= 4 and 
                            clean_word not in exclude_words and
                            not clean_word.isdigit() and
                            clean_word.isalpha()):
                            
                            # Additional check: if it's a known medicine or has medicine suffix
                            if (clean_word in known_medicines or 
                                any(clean_word.endswith(suffix) for suffix in medicine_suffixes)):
                                medicines_found.add(clean_word.title())
        
        # Final filtering and formatting
        final_medicines = []
        for medicine in medicines_found:
            # Skip single letters or very short words
            if len(medicine) < 4:
                continue
                
            # Skip if it's clearly not a medicine
            medicine_lower = medicine.lower()
            if medicine_lower in exclude_words:
                continue
                
            # Add to final list
            final_medicines.append(medicine)
        
        # Sort alphabetically and return max 8 medicines
        return sorted(list(set(final_medicines)))[:8]