from langchain_community.llms import Ollama

# Load the LLaMA model (Mistral via Ollama)
llm = Ollama(model="llama3.2")

# Generate a topic
def generate_topic_triple():
    topic_prompt = ("Generate a unique and engaging topic for a speaker to talk about. "
                    "Give only the topic. Do not include any extra words.")
    topic_response = llm.invoke(topic_prompt)
    generated_topic = topic_response.strip()
    print("\nGenerated Topic:", generated_topic)
    return generated_topic

# Generate relevant distractor words
def generate_distractor_words(topic):
    distractor_prompt = f"Generate 5 distractor words related to the topic '{topic}'.\nGive only the words, each on a new line. Do not include numbering"
    distractor_response = llm.invoke(distractor_prompt)
    distractor_words = [word.strip() for word in distractor_response.strip().split("\n")]
    print("\nDistractor Words:", distractor_words)
    return distractor_words

# Function to get AI feedback
def get_llama_feedback_triple(topic, transcription, mfcc_features, distractor_count):
    mfcc_summary = ", ".join([f"{feat:.2f}" for feat in mfcc_features])
    feedback_prompt = f"""
    Evaluate the following speech based on relevance to the topic, pronunciation, coherence, articulation, and grammar.
    Additionally, consider the MFCC features and the number of distractor words used.
    GIVE THE SPEAKER A SCORE OUT OF 100
    
    **Topic:** {topic}
    
    **Transcription:** {transcription}
    
    **MFCC Features (mean values):** {mfcc_summary}
    
    **Distractor Words Used:** {distractor_count}
    
    Feedback should include:
    - Relevance to the topic(score)
    - Clarity and coherence(score)
    - Pronunciation and articulation(score)
    - Grammar and fluency(score)
    - How well the speaker avoided distractor words(score)
    - Overall effectiveness(Overall effectiveness should be the cumulative score of the above scores)
    
    Don't say you are evaluating based on transcription"""
    
    feedback_response = llm.invoke(feedback_prompt)
    print("\n=== LLaMA AI Feedback ===")
    print(feedback_response)
    clean_output = feedback_response.replace("*", "")


    return clean_output
