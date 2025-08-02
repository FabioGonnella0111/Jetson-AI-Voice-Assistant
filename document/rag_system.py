import os
import time
import logging
import numpy as np
import gc
import torch
import re
import json
import hashlib
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

# Configurazione logging ottimizzata
logging.basicConfig(
    level=logging.INFO,  # Ridotto da DEBUG per performance
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_optimized.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Struttura dati per i risultati di ricerca"""
    idx: int
    score: float
    text: str
    metadata: Optional[Dict] = None

class TextProcessor:
    """Processore di testo migliorato con chunking intelligente"""
    
    @staticmethod
    def smart_chunk_text(text: str, max_chunk_size: int = 300, overlap: int = 50) -> List[Tuple[str, Dict]]:
        """
        Chunking intelligente che rispetta i confini delle frasi e paragrafi
        """
        # Prima divisione per paragrafi
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        for para_idx, paragraph in enumerate(paragraphs):
            # Dividi il paragrafo in frasi
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]
            
            current_chunk = ""
            current_sentences = []
            
            for sent_idx, sentence in enumerate(sentences):
                # Se aggiungere questa frase supera la dimensione massima
                if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                    # Salva il chunk corrente
                    metadata = {
                        'paragraph_idx': para_idx,
                        'sentence_range': (sent_idx - len(current_sentences), sent_idx - 1),
                        'chunk_type': 'sentence_group'
                    }
                    chunks.append((current_chunk.strip(), metadata))
                    
                    # Inizia nuovo chunk con overlap
                    if overlap > 0 and current_sentences:
                        overlap_sentences = current_sentences[-min(2, len(current_sentences)):]
                        current_chunk = ' '.join(overlap_sentences) + ' ' + sentence
                        current_sentences = overlap_sentences + [sentence]
                    else:
                        current_chunk = sentence
                        current_sentences = [sentence]
                else:
                    # Aggiungi la frase al chunk corrente
                    if current_chunk:
                        current_chunk += ' ' + sentence
                    else:
                        current_chunk = sentence
                    current_sentences.append(sentence)
            
            # Aggiungi l'ultimo chunk se non vuoto
            if current_chunk.strip():
                metadata = {
                    'paragraph_idx': para_idx,
                    'sentence_range': (len(sentences) - len(current_sentences), len(sentences) - 1),
                    'chunk_type': 'sentence_group'
                }
                chunks.append((current_chunk.strip(), metadata))
        
        return chunks

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocessing del testo per migliorare la qualit  degli embeddings"""
        # Normalizza spazi bianchi
        text = re.sub(r'\s+', ' ', text)
        # Rimuovi caratteri di controllo
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        # Normalizza punteggiatura
        text = re.sub(r'([.!?])\s*([.!?])+', r'\1', text)
        return text.strip()

class RagSystem:
    """Sistema RAG ottimizzato per performance e qualit """
    
    def __init__(self,
                 txt_dir: str = "uploads",
                 emb_file: str = "embeddings.npz",
                 metadata_file: str = "metadata.json",
                 model_name: str = './models/all-MiniLM-L6-v2',
                 reindex: bool = False,
                 chunk_size: int = 300,
                 chunk_overlap: int = 50):
        
        logger.info(f"Inizializzazione RagSystem ottimizzato")
        start_time = time.time()

        self.txt_dir = txt_dir
        self.emb_file = emb_file
        self.metadata_file = metadata_file
        self.reindex = reindex
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Cache per evitare ricalcoli
        self._embeddings_cache = None
        self._chunks_cache = None
        self._metadata_cache = None

        # Ottimizzazioni per dispositivi low-end
        self._setup_environment()
        
        # Carica modello
        self.model = self._load_model(model_name)
        
        logger.info(f"Inizializzazione completata in {time.time() - start_time:.2f} secondi")

    def _setup_environment(self):
        """Configura l'ambiente per performance ottimali"""
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = '2'  # Limita thread per Jetson Nano
        
        if torch.cuda.is_available():
            logger.info("CUDA disponibile ma usando CPU per stabilit ")
        
        # Gestione memoria PyTorch
        torch.set_num_threads(2)

    def _load_model(self, model_path: str) -> SentenceTransformer:
        """Carica il modello con gestione errori migliorata"""
        model_path = os.path.abspath(model_path)
        logger.info(f"Caricamento modello: {model_path}")
        
        try:
            model = SentenceTransformer(model_path, device='cpu')
            model.eval()
            
            # Test rapido del modello
            with torch.no_grad():
                test_embedding = model.encode(["test"], convert_to_numpy=True)
                logger.info(f"Modello caricato - dimensione embedding: {test_embedding.shape[1]}")
            
            return model
            
        except Exception as e:
            logger.error(f"Errore caricamento modello: {e}")
            raise FileNotFoundError(
                f"Impossibile caricare il modello da '{model_path}'. "
                f"Verifica che sia presente e contenga tutti i file necessari."
            )

    def _compute_file_hash(self) -> str:
        """Calcola hash dei file per verificare se serve reindicizzazione"""
        if not os.path.exists(self.txt_dir):
            return ""
        
        txt_files = sorted([f for f in os.listdir(self.txt_dir) if f.lower().endswith('.txt')])
        hash_input = ""
        
        for fname in txt_files:
            fpath = os.path.join(self.txt_dir, fname)
            mtime = os.path.getmtime(fpath)
            hash_input += f"{fname}:{mtime}:"
        
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _read_and_process_data(self) -> Tuple[List[str], List[Dict]]:
        """Legge e processa i dati con chunking intelligente"""
        logger.info(f"Lettura e processamento da: {self.txt_dir}")
        start_time = time.time()

        if not os.path.exists(self.txt_dir) or not os.path.isdir(self.txt_dir):
            raise FileNotFoundError(f"Directory '{self.txt_dir}' non trovata!")

        txt_files = sorted([f for f in os.listdir(self.txt_dir) if f.lower().endswith('.txt')])
        if not txt_files:
            raise FileNotFoundError(f"Nessun file .txt trovato in {self.txt_dir}")

        all_chunks = []
        all_metadata = []
        
        processor = TextProcessor()
        
        for file_idx, fname in enumerate(txt_files):
            logger.debug(f"Processando file: {fname}")
            fpath = os.path.join(self.txt_dir, fname)
            
            with open(fpath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # Preprocessing
            clean_text = processor.preprocess_text(raw_text)
            
            # Chunking intelligente
            chunks_with_meta = processor.smart_chunk_text(
                clean_text, 
                max_chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            
            # Aggiungi metadata del file
            for chunk_text, chunk_meta in chunks_with_meta:
                chunk_meta.update({
                    'file_name': fname,
                    'file_idx': file_idx,
                    'chunk_id': len(all_chunks)
                })
                all_chunks.append(chunk_text)
                all_metadata.append(chunk_meta)

        logger.info(f"Processati {len(all_chunks)} chunks da {len(txt_files)} file in {time.time() - start_time:.2f} secondi")
        return all_chunks, all_metadata

    def _should_reindex(self) -> bool:
        """Determina se   necessaria la reindicizzazione"""
        if self.reindex:
            return True
            
        if not os.path.exists(self.emb_file) or not os.path.exists(self.metadata_file):
            return True
        
        # Controlla se i file sono cambiati
        current_hash = self._compute_file_hash()
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                stored_data = json.load(f)
                stored_hash = stored_data.get('file_hash', '')
                return current_hash != stored_hash
        except:
            return True

    def index_database(self) -> np.ndarray:
        """Indicizza il database con gestione cache e validazione"""
        logger.info("Inizio indicizzazione database")
        start_time = time.time()

        # Leggi e processa i dati
        chunks, metadata = self._read_and_process_data()
        
        if not chunks:
            raise ValueError("Nessun chunk di testo da indicizzare!")

        # Genera embeddings in batch
        embeddings_list = []
        batch_size = 32  # Ridotto per dispositivi low-end
        
        logger.info(f"Generazione embeddings per {len(chunks)} chunks")
        
        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                try:
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=batch_size,
                        normalize_embeddings=True
                    )
                    embeddings_list.append(batch_embeddings)
                    
                    # Log progresso ogni 50 batch
                    if (i // batch_size) % 50 == 0:
                        logger.debug(f"Processati {i + len(batch)}/{len(chunks)} chunks")
                    
                    # Garbage collection periodico
                    if i % (batch_size * 10) == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Errore nel batch {i}-{i+batch_size}: {e}")
                    raise

        # Combina tutti gli embeddings
        embeddings = np.vstack(embeddings_list)
        
        # Salva embeddings e metadata
        np.savez_compressed(self.emb_file, embeddings=embeddings)
        
        # Salva metadata con hash dei file
        metadata_to_save = {
            'chunks': chunks,
            'metadata': metadata,
            'file_hash': self._compute_file_hash(),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_dim': embeddings.shape[1],
            'total_chunks': len(chunks),
            'created_at': time.time()
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)

        # Cache in memoria
        self._embeddings_cache = embeddings
        self._chunks_cache = chunks
        self._metadata_cache = metadata

        logger.info(f"Indicizzazione completata in {time.time() - start_time:.2f} secondi")
        logger.info(f"Shape embeddings: {embeddings.shape}")
        
        return embeddings

    def _load_cached_data(self) -> Tuple[np.ndarray, List[str], List[Dict]]:
        """Carica dati dalla cache o dai file"""
        if (self._embeddings_cache is not None and 
            self._chunks_cache is not None and 
            self._metadata_cache is not None):
            return self._embeddings_cache, self._chunks_cache, self._metadata_cache

        # Carica da file
        if not os.path.exists(self.emb_file) or not os.path.exists(self.metadata_file):
            raise FileNotFoundError("File di cache non trovati. Esegui prima index_database()")

        # Carica embeddings
        emb_data = np.load(self.emb_file)
        embeddings = emb_data['embeddings']

        # Carica metadata
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            stored_data = json.load(f)
        
        chunks = stored_data['chunks']
        metadata = stored_data['metadata']

        # Cache in memoria
        self._embeddings_cache = embeddings
        self._chunks_cache = chunks
        self._metadata_cache = metadata

        return embeddings, chunks, metadata

    def search(self, query: str, top_k: int = 10, score_threshold: float = 0.3) -> List[SearchResult]:
        """
        Ricerca migliorata con filtering e ranking
        """
        logger.debug(f"Ricerca per: '{query}' (top_k={top_k}, threshold={score_threshold})")
        
        # Carica dati
        embeddings, chunks, metadata = self._load_cached_data()
        
        # Genera embedding della query
        with torch.no_grad():
            query_emb = self.model.encode(
                [query], 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )[0]

        # Calcola similarit 
        similarities = np.dot(embeddings, query_emb)
        
        # Filtra per soglia minima
        valid_indices = np.where(similarities >= score_threshold)[0]
        
        if len(valid_indices) == 0:
            logger.warning(f"Nessun risultato sopra la soglia {score_threshold}")
            # Prendi comunque i migliori 3 risultati
            top_k = min(3, len(similarities))
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
        else:
            # Prendi i migliori tra quelli validi
            valid_similarities = similarities[valid_indices]
            top_k = min(top_k, len(valid_indices))
            relative_top_indices = np.argpartition(-valid_similarities, top_k)[:top_k]
            top_indices = valid_indices[relative_top_indices]

        # Crea risultati ordinati
        results = []
        for idx in top_indices:
            result = SearchResult(
                idx=int(idx),
                score=float(similarities[idx]),
                text=chunks[idx],
                metadata=metadata[idx] if idx < len(metadata) else None
            )
            results.append(result)

        # Ordina per score decrescente
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results

    def get_context(self, query: str, max_context_length: int = 1500, top_k: int = 10) -> str:
        """
        Genera contesto ottimizzato per LLM con deduplicazione intelligente
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return "Nessun contenuto rilevante trovato."

        # Deduplicazione intelligente
        seen_texts = set()
        unique_results = []
        
        for result in results:
            # Usa un hash del testo normalizzato per evitare duplicati quasi-identici
            normalized = re.sub(r'\s+', ' ', result.text.lower().strip())
            text_hash = hashlib.md5(normalized.encode()).hexdigest()
            
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_results.append(result)

        # Costruisci contesto con priorit  ai risultati migliori
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(unique_results):
            # Aggiungi separatore informativo
            header = f"\n--- Risultato {i+1} (score: {result.score:.3f}) ---\n"
            content = result.text
            
            estimated_length = len(header) + len(content)
            
            if current_length + estimated_length > max_context_length and context_parts:
                break
                
            context_parts.append(header + content)
            current_length += estimated_length

        context = "\n".join(context_parts)
        
        logger.info(f"Contesto generato: {len(unique_results)} chunks unici, {len(context)} caratteri")
        return context

    def run(self, query: str, top_k: int = 5, return_context: bool = True, verbose: bool = False) -> Union[str, List[SearchResult]]:
        """
        Esecuzione principale con opzioni flessibili
        """
        logger.info(f"Esecuzione RAG - Query: '{query}', top_k: {top_k}")
        start_time = time.time()

        try:
            # Verifica se serve reindicizzazione
            if self._should_reindex():
                logger.info("Reindicizzazione necessaria")
                self.index_database()

            # Esegui ricerca
            results = self.search(query, top_k=top_k)
            
            if verbose:
                logger.info("=== RISULTATI RICERCA ===")
                for i, result in enumerate(results):
                    logger.info(f"[{i+1}] Score: {result.score:.3f}")
                    logger.info(f"Testo: {result.text[:100]}...")
                    if result.metadata:
                        logger.info(f"Metadata: {result.metadata}")

            execution_time = time.time() - start_time
            logger.info(f"Ricerca completata in {execution_time:.2f} secondi")

            if return_context:
                return self.get_context(query, top_k=len(results))
            else:
                return results

        except Exception as e:
            logger.error(f"Errore durante l'esecuzione: {e}")
            raise

    def get_stats(self) -> Dict:
        """Restituisce statistiche del sistema"""
        try:
            embeddings, chunks, metadata = self._load_cached_data()
            
            return {
                'total_chunks': len(chunks),
                'embedding_dimension': embeddings.shape[1],
                'avg_chunk_length': np.mean([len(chunk) for chunk in chunks]),
                'cache_status': 'loaded' if self._embeddings_cache is not None else 'not_loaded',
                'files_indexed': len(set(m.get('file_name', '') for m in metadata)),
                'index_exists': os.path.exists(self.emb_file)
            }
        except:
            return {'error': 'Dati non disponibili'}

    def clear_cache(self):
        """Pulisce la cache in memoria"""
        self._embeddings_cache = None
        self._chunks_cache = None
        self._metadata_cache = None
        gc.collect()
        logger.info("Cache pulita")

    def similarity_search_with_filter(self, 
                                    query: str, 
                                    file_filter: Optional[str] = None,
                                    metadata_filter: Optional[Dict] = None,
                                    top_k: int = 10) -> List[SearchResult]:
        """
        Ricerca con filtri avanzati su file o metadata
        """
        results = self.search(query, top_k=top_k * 2)  # Prendi pi  risultati per filtrare
        
        filtered_results = []
        for result in results:
            # Filtro per nome file
            if file_filter and result.metadata:
                if file_filter.lower() not in result.metadata.get('file_name', '').lower():
                    continue
            
            # Filtro per metadata
            if metadata_filter and result.metadata:
                match = True
                for key, value in metadata_filter.items():
                    if result.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            filtered_results.append(result)
            
            if len(filtered_results) >= top_k:
                break

        return filtered_results

# Esempio di utilizzo ottimizzato
if __name__ == "__main__":
    # Inizializza sistema
    rag = RagSystem(
        txt_dir="uploads",
        chunk_size=250,      # Chunk pi  piccoli per maggiore precisione
        chunk_overlap=30,    # Overlap ridotto per performance
        reindex=False        # Usa cache se disponibile
    )
    
    # Mostra statistiche
    stats = rag.get_stats()
    print("=== STATISTICHE SISTEMA ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Esempio di ricerca
    query = "solar collector area challenger vehicles"
    
    print(f"\n=== RICERCA: '{query}' ===")
    context = rag.run(query, top_k=5, verbose=True)
    print(f"\nContesto generato ({len(context)} caratteri):")
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # Esempio con filtri
    print(f"\n=== RICERCA CON FILTRI ===")
    filtered_results = rag.similarity_search_with_filter(
        query="energy storage",
        file_filter="regolamento",  # Solo dal file regolamento
        top_k=3
    )
    
    for i, result in enumerate(filtered_results):
        print(f"[{i+1}] Score: {result.score:.3f} - File: {result.metadata.get('file_name', 'N/A')}")
        print(f"Testo: {result.text[:100]}...")