from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# from langchain_ollama import ChatOllama

load_dotenv()


def main():
    information = """
        Alan Mathison Turing (23 June 1912 – 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist.[6] He was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer.[7][8][9] Turing is widely considered to be the father of theoretical computer science.[10]
        
        Born in London, Turing was raised in southern England. He graduated from King's College, Cambridge, and in 1938, earned a doctorate degree from Princeton University. During World War II, Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence. He led Hut 8, the section responsible for German naval cryptanalysis. Turing devised techniques for speeding the breaking of German ciphers, including improvements to the pre-war Polish bomba method, an electromechanical machine that could find settings for the Enigma machine. He played a crucial role in cracking intercepted messages that enabled the Allies to defeat the Axis powers in the Battle of the Atlantic and other engagements.[11][12]
        
        After the war, Turing worked at the National Physical Laboratory, where he designed the Automatic Computing Engine, one of the first designs for a stored-program computer. In 1948, Turing joined Max Newman's Computing Machine Laboratory at the University of Manchester, where he contributed to the development of early Manchester computers[13] and became interested in mathematical biology. Turing wrote on the chemical basis of morphogenesis[14][1] and predicted oscillating chemical reactions such as the Belousov–Zhabotinsky reaction, first observed in the 1960s. Despite these accomplishments, he was never fully recognised during his lifetime because much of his work was covered by the Official Secrets Act.[15]
        
        In 1952, Turing was prosecuted for homosexual acts. He accepted hormone treatment, a procedure commonly referred to as chemical castration, as an alternative to prison. Turing died on 7 June 1954, aged 41, from cyanide poisoning. An inquest determined his death as suicide, but the evidence is also consistent with accidental poisoning.[5] Following a campaign in 2009, British prime minister Gordon Brown made an official public apology for "the appalling way [Turing] was treated". Queen Elizabeth II granted a pardon in 2013. The term "Alan Turing law" is used informally to refer to a 2017 law in the UK that retroactively pardoned men cautioned or convicted under historical legislation that outlawed homosexual acts.[16]
        
        Turing left an extensive legacy in mathematics and computing which has become widely recognised with statues and many things named after him, including an annual award for computing innovation. His portrait appears on the Bank of England £50 note, first released on 23 June 2021 to coincide with his birthday. The audience vote in a 2019 BBC series named Turing the greatest scientist of the 20th century.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    llm = ChatOpenAI(temperature=0, model="gpt-5")
    # (pipe operator in LangChain Expression Language - LCEL)
    chain = summary_prompt_template | llm
    #
    # formatted_prompt = summary_prompt_template.invoke({"information": information})
    # response = llm.invoke(formatted_prompt)
    #
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
#
# 1) Short summary:
# Alan Turing (1912–1954) was an English mathematician and pioneer of computer science whose concept of the Turing machine formalized computation and algorithms. During World War II he led Hut 8 at Bletchley Park, devising methods to break German naval Enigma, crucial to Allied success. After the war he designed early stored‑program computer architectures and made influential contributions to mathematical biology. Persecuted in 1952 for his homosexuality, he died in 1954; he has since been formally pardoned and widely celebrated for his legacy.
#
# 2) Two interesting facts:
# - In 1952 he published a theory of morphogenesis that predicted oscillating chemical reactions, decades before the Belousov–Zhabotinsky reaction was observed.
# - His portrait appears on the Bank of England £50 note, released on his birthday, 23 June 2021.
