from llama_index.core import (
    SimpleDirectoryReader, 
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import uuid 
import copy
from llama_index.core.schema import TextNode
import pickle

def init_settings(llm, embed_model):
    Settings.llm = llm
    Settings.embed_model = embed_model

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_pkl(path, obj2save):
    with open(path, "wb") as f:
        pickle.dump(obj2save, f)

def load_base_nodes(input_files=None, input_dir=None, min_length=50):
    """Reads files / dir and returns nodes with a min length"""
    if (input_files and input_dir) or (not input_files and not input_dir):
        raise ValueError("Provide either input_files or input_dir, but not both")
    
    reader = SimpleDirectoryReader(input_files=input_files) if input_files else SimpleDirectoryReader(input_dir=input_dir)
    
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024)
    base_nodes = node_parser.get_nodes_from_documents(documents)
    
    filtered_nodes = []
    for base_node in base_nodes:
        if len(base_node.text) > min_length: 
            base_node.node_id = str(uuid.uuid4())
            filtered_nodes.append(base_node)
    
    return filtered_nodes

def extract_metadata_from_nodes(names, extractors, base_nodes, s=0, e=None, output_save=None, tracker=None):
    """
    Extracts metadata from a list of nodes using the provided extractors
    
    Notes:
    - You can use multiple extractors, but if using multiple extractors, you will not be able to use node embeddings unless the `QAExtractor` is the only one used.
    """
    
    metadata_dict = []

    # CodeCarbone tracker
    if tracker:
        tracker.start()
        
    for name, extractor in zip(names, extractors):
        metadata_dict.extend(extractor.extract(base_nodes[slice(s, e)]))
        if output_save:
            save_pkl(output_save, metadata_dict)
            print(f"Metadata have been saved to {output_save}")

    if tracker:
        tracker.stop()

    return metadata_dict

def get_nodes_embeddings(base_nodes, metadata_dict, s=0, e=None, output_save=None, tracker=None):
    """Only works with qa metadata for the moment"""
    slice_nodes = copy.deepcopy(base_nodes)
    slice_nodes = slice_nodes[s:e]
    
    all_nodes = []
    
    qa_key = "questions_this_excerpt_can_answer"

    # CodeCarbone tracker
    if tracker:
        tracker.start()
        
    for idx, base_node in enumerate(slice_nodes):
        metadata_qa = metadata_dict[idx]
        
        node_metadata = base_node.metadata.copy()
    
        questions = metadata_qa.get(qa_key, [])
    
        node_metadata["questions"] = questions
        
        text_node = TextNode(
            id=base_node.node_id,
            text=f"{questions}{base_node.get_text()}",  
            metadata=node_metadata  
        )
    
        all_nodes.append(text_node)
    
    texts = [node.text for node in all_nodes]
    embeddings = embed_model.get_text_embedding_batch(texts)  
    
    for i, node in enumerate(all_nodes):
        node.embedding = embeddings[i]
    
    if tracker:
        tracker.stop()
    
    if output_save:
        save_pkl(output_save, all_nodes)

    return all_nodes