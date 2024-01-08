from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path
import os
from langchain.document_loaders import UnstructuredURLLoader
import faiss


base_dir = Path(__file__).resolve().parent
DB_FAISS_PATH = index_path = os.path.join(base_dir, 'vectorstore', 'db_faiss')


# import session_info

# session_info.show()

urls = [
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/prevention",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/alert",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/escape",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/what-to-do-in-the-event-of-a-fire",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/what-to-do-after-a-fire",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/burns-first-aid",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/barbecue-safety",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/bedtime-check",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/bonfire-night-safety",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/candle-safety",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/carbon-monoxide",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/celebrate-safely",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/cooking-fire-safety",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/electrical-fire-safety",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/ebikes-and-escooters",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/electric-blankets",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/emollient-skin-products",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/escape-routes",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/flooding",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/flying-lanterns",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/fuel-storage",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/grassland-fires",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/halloween",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/high-rise-living",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/hoarding",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/batteries",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/oxygen-equipment-safety",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/referral-and-contact",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/rented-homes",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/smoking-safety",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/student-safety",
    "https://humbersidefire.gov.uk/your-safety/safety-in-the-home-advice/wheelie-bins",
    "https://humbersidefire.gov.uk/your-safety/our-work-in-the-community",
    "https://humbersidefire.gov.uk/your-safety/our-work-in-the-community/fire-cadets",
    "https://humbersidefire.gov.uk/your-safety/our-work-in-the-community/community-safety-volunteers",
    "https://humbersidefire.gov.uk/your-safety/our-work-in-the-community/firefli-research-project",
    "https://humbersidefire.gov.uk/your-safety/our-work-in-the-community/firestoppers",
    "https://humbersidefire.gov.uk/your-safety/our-work-in-the-community/staywise-online-educational-resources",
    "https://humbersidefire.gov.uk/your-safety/unwanted-fire-signals",
    "https://humbersidefire.gov.uk/your-safety/unwanted-fire-signals/non-attendance-and-cost-recovery",
    "https://humbersidefire.gov.uk/your-safety/unwanted-fire-signals/call-out-charges",
    "https://humbersidefire.gov.uk/your-safety/unwanted-fire-signals/exempt-businesses-and-heritage-buildings",
    "https://humbersidefire.gov.uk/your-safety/unwanted-fire-signals/common-causes-of-alarm-activations",
    "https://humbersidefire.gov.uk/your-safety/unwanted-fire-signals/fire-alarm-system-checklists",
    "https://humbersidefire.gov.uk/your-safety/business-safety",
    "https://humbersidefire.gov.uk/your-safety/business-safety/fire-risk-assessment-guidance",
    "https://humbersidefire.gov.uk/your-safety/business-safety/fire-safety-on-the-farm",
    "https://humbersidefire.gov.uk/your-safety/business-safety/inspection-and-enforcement",
    "https://humbersidefire.gov.uk/your-safety/business-safety/how-to-reduce-deliberate-fires",
    "https://humbersidefire.gov.uk/your-safety/business-safety/living-above-a-business",
    "https://humbersidefire.gov.uk/your-safety/business-safety/hmos-houses-in-multiple-occupation",
    "https://humbersidefire.gov.uk/your-safety/business-safety/sprinklers",
    "https://humbersidefire.gov.uk/your-safety/business-safety/residential-care-home-guides",
    "https://humbersidefire.gov.uk/your-safety/business-safety/the-licensing-act",
    "https://humbersidefire.gov.uk/your-safety/business-safety/contact-the-business-safety-team",
    "https://humbersidefire.gov.uk/your-safety/business-safety/fser",
    "https://humbersidefire.gov.uk/your-safety/business-safety/cost-of-living-advice-for-businesses-and-building-owners",
    "https://humbersidefire.gov.uk/your-safety/road-safety",
    "https://humbersidefire.gov.uk/your-safety/water-safety-and-drowning-prevention",
    "https://humbersidefire.gov.uk/contact-us",
    "https://humbersidefire.gov.uk/about-us/principles-and-behaviours",
    "https://nfcc.org.uk/our-services/building-safety",
    "https://www.legislation.gov.uk/ukpga/2004/21/contents",
    "https://staywise.co.uk/public",
    "https://firekills.campaign.gov.uk"
]



# Create vector database
def create_vector_db():
    load_dotenv()
    loader = UnstructuredURLLoader(urls=urls)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(documents)
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local(DB_FAISS_PATH)

def count_vectors_in_faiss_index(index_path):
    # Load the index from the file
    index = faiss.read_index(index_path)
    return index.ntotal




if __name__ == "__main__":
    create_vector_db()
    # DB_FAISS_INDEX = index_path = os.path.join(base_dir, 'vectorstore', 'db_faiss', 'index.faiss')
    # num_vectors = count_vectors_in_faiss_index(DB_FAISS_INDEX)
    # print(f"The number of vectors in the index: {num_vectors}")

