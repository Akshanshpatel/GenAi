# This file will take str input(essay) and perform parallelized workflow via langraph in order to 
# produce a summarized feedback and score out of 10 for essay based on clarity of thought, concept
#  and language(words used)

from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel,Field

load_dotenv()

class essayState(TypedDict):
    essay:str
    cot_feedback:str
    concept_feedback:str
    lang_feedback:str
    final_feedback:str
    # We can also create a list instead of below 3 variables but that requires understanding reducers
    # for merging marks in list while parralel operation
    cot_marks:int
    concept_marks:int
    lang_marks:int
    marks:float

class evaluationSchema(BaseModel):
    feedback:str = Field(description='Detailed feedback on the essay')
    score:int = Field(description='Score out of 10',ge=0,le=10)

model=ChatGroq(model="openai/gpt-oss-120b")

str_model=model.with_structured_output(evaluationSchema)


essay_input= """Life can only be understood backwards; but it must be lived forwards." — Søren Kierkegaard 


Kierkegaard’s words themselves contain a quiet paradox: we must move forward through life even though its meaning often becomes clear only in retrospect. Perhaps this tension between living and understanding explains why contradiction finds such a natural place in human language. 

We often use words that seem to contradict each other — a "deafening silence," a "living death," a "cruel kindness." At first, these sound like mistakes. But they are not. They are small, honest snapshots of life itself, because real life is rarely simple or one-sided. An oxymoron joins two opposite ideas in a single phrase, and in doing so, it captures something plain language cannot: the fact that joy and sorrow, freedom and discipline, love and pain often exist together, not apart. To say that oxymorons reflect the ironies of life, then, is to say that contradiction is not a flaw in human existence — it is a basic part of it. Poets and philosophers have always understood this, even when logic resists it: that truth is often paradoxical, and that wisdom lies in holding two opposite truths together, without rejecting either one. 

The Oxymoron as a Mirror of the Human Condition 

Human life is seldom lived in straight lines. We love the people who hurt us most; we find our deepest freedom in discipline; we grow through suffering we would never wish upon ourselves. The Buddhist concept of dukkha i.e suffering woven into the very fabric of existence, sits beside the equally Buddhist insistence that liberation, nirvana, is found not by escaping the world but by seeing through it clearly. This is oxymoronic at its core: bondage and liberation share the same address. 

Rabindranath Tagore captured this when he wrote, "I have on my table a violin string. It is free." A string, by definition, is bound — tied down, restrained — and yet it is precisely this restraint that allows it to produce music, to become "free" in the only sense that matters for a string: expressive, alive, purposeful. Constraint and freedom, ordinarily antonyms, become collaborators. This is the oxymoron functioning not as decoration but as diagnosis. 

Consider the phrase "organised chaos," often used to describe an Indian railway station, a wedding, or even the country's democracy itself. On the surface, it is nonsensical — chaos, by definition, resists organisation. Yet anyone who has watched a marketplace in Chandni Chowk function with uncanny efficiency amid apparent disorder knows that the phrase is not a contradiction but a compression of reality. India itself has often been called "a functioning anarchy," a phrase associated with the economist John Kenneth Galbraith. It captures the paradox of a nation that should not work on paper — with its scale, diversity, and inequality, and yet does, however imperfectly, function. 

Irony as the Texture of History and Politics 

History rarely rewards the expectation of straightforward outcomes. Mahatma Gandhi's philosophy of ahimsa i.e., non-violence, proved to be one of the most powerful weapons ever wielded against an empire. The world's most disciplined army of colonial administrators was undone not by superior firepower but by the "militant peace" of unarmed protestors willing to be beaten and not retaliate. Gandhi himself embraced the paradox openly: "In a gentle way, you can shake the world." Gentleness and world-shaking do not belong in the same sentence — until they do, and history is changed. 

Nelson Mandela's twenty-seven years in prison did not produce a broken man seeking revenge but a "victorious prisoner" whose incarceration became the moral capital for reconciliation rather than retribution. He walked out of Robben Island not diminished but enlarged, a bitter irony for his captors, who had hoped imprisonment would silence him, and instead it made him unsilenceable. 

Modern geopolitics, too, thrives on oxymoronic description. "Cold War", a war fought without direct battles between its principal antagonists — defined half a century of global history. Nuclear deterrence rests on what strategists frankly call "mutually assured destruction," memorably abbreviated to the acronym MAD: the doctrine that peace is preserved by each side's terrifying capacity to destroy the other. Peace, here, is not the absence of violence but it's perfect, symmetrical threat. As George Orwell foresaw in a different context, the language of power often requires exactly such doublespeak: "War is Peace." What Orwell offered as dystopian warning, statecraft has, in a sobered and rationalised form, actually practised. 

Even India’s economic journey bears an oxymoronic imprint. As some scholars describe it, India is an “underdeveloped superpower”, a civilisation endowed with millennia of cultural and intellectual capital and a formidable nuclear, space and diplomatic presence, yet still grappling with widespread poverty, malnutrition and inequality. Economic Surveys and Human Development Reports repeatedly capture this paradox: rising GDP alongside persistent deprivation and economic growth without commensurate employment generation, the phenomenon of “jobless growth.” 

Literature and Philosophy: Making Peace with Paradox 

Literature has always understood that truth wears contradiction as its finest garment. Shakespeare filled his plays with oxymorons precisely because his characters live in morally ambiguous worlds. Romeo, lovesick and confused, calls love "a smoke made with the fume of sighs," "a madness most discreet," describing his feelings as "cold fire, sick health." He is not being clumsy; he is being accurate. Adolescent love is a "sweet sorrow," as Juliet later says of parting from Romeo , and anyone who has loved and lost knows that no single, unconflicted adjective could ever do that feeling justice. 

The Bhagavad Gita, arguably the philosophical bedrock of Indian ethical thought, is itself built on a magnificent paradox: Krishna counsels Arjuna to act, and act decisively in battle, while simultaneously renouncing all attachment to the fruits of that action. This is nishkama karma — desireless action — labour undertaken with full intensity but without the ego's craving for reward. Effort without attachment; engagement without entanglement. It is, in essence, a spiritually sanctioned oxymoron, and it has guided the ethical imagination of a civilisation for millennia. 

Albert Camus, writing from a very different philosophical shore, arrived at a strikingly similar structure of thought. Confronting what he called the "absurd" — the human need for meaning colliding with a universe that offers none — Camus did not counsel despair but rebellion: "One must imagine Sisyphus happy." A man condemned eternally to push a boulder uphill only to watch it roll down again, and yet — happy. Meaningless labour, joyfully embraced. The absurd hero is, definitionally, an oxymoron: dignity found in futility. 

Governance, Development, and the Oxymorons of Our Time 

The vocabulary of contemporary governance is saturated with such paradoxes, often without anyone noticing. "Sustainable development" itself is a term some early ecologists considered close to oxymoronic — development, historically, has meant extraction, consumption, and growth, while sustainability demands restraint and preservation. That the term has become the organising principle of global policy, from the Sustainable Development Goals to India's own climate commitments, shows how a society under pressure learns to hold two demands — growth and restraint — in permanent, productive tension rather than resolving them falsely in favour of one. 

Similarly, India's constitutional vision embeds oxymoron at a structural level: "positive discrimination," the term often used for affirmative action and reservation policy, asks discrimination — ordinarily a corrosive, unjust act — to serve the cause of justice and equality. Ambedkar himself was acutely aware of this tension when he warned the Constituent Assembly of India entering "a life of contradictions" in 1950 — political equality alongside social and economic inequality. He did not resolve the contradiction rhetorically; he built institutions to manage it over time. Democracy, similarly, is often called "government by discussion," a phrase that presumes an entire nation of contradictory opinions can somehow crystallise into a single legitimate decision — dissent transformed into consent, plurality distilled into unity. "Unity in diversity," India's own civilisational motto, is oxymoronic in exactly this productive sense. 

When Contradictions Become Dangerous 

Yet not every oxymoron contains wisdom. There is a danger in romanticising contradiction and suffering. Poverty should not be celebrated because it supposedly produces resilience; exploitation does not become desirable because struggle can build character. Avoidable suffering remains an injustice demanding correction. 

Language can also conceal brutality beneath attractive contradictions. Expressions such as “benevolent dictatorship” may sanitise authoritarianism; “honour killing” attaches dignity to murder; and “collateral damage” can distance decision-makers from human suffering. George Orwell's chilling “War is Peace” reminds us that contradictions can be manipulated to obscure truth rather than reveal it. 

Wisdom, therefore, is not the passive acceptance of every contradiction. It requires distinguishing productive tensions from moral evasions. Legitimate competing values must be reconciled through reason, empathy, rights, proportionality and accountability. Where contradiction masks injustice, a clear moral position becomes necessary. 

Socrates' philosophy offers an appropriate reminder: wisdom begins with recognising the limits of one's knowledge. Intellectual maturity does not always demand certainty; sometimes it requires the humility to recognise that two legitimate truths can coexist.

Conclusion 

Oxymorons reflect the ironies of life because existence is woven from opposing strands. Joy acquires depth through sorrow, courage emerges in the presence of fear, freedom survives through responsibility, and hope becomes meaningful amid adversity. The opposites of life do not always destroy each other; often, they define, correct and complete one another. 

A river reaches the sea not by following a perfectly straight path but by negotiating rocks, slopes and changing landscapes. Human civilisation progresses in much the same manner—through conflict and reconciliation, error and correction, destruction and renewal. The aim of wisdom is not to eliminate every contradiction, for that may be impossible. It is to distinguish between contradictions that enrich life and those that perpetuate injustice. 

Ultimately, the greatest oxymoron may be the human condition itself: we are fragile beings capable of extraordinary resilience, imperfect individuals pursuing perfection, and temporary lives seeking enduring meaning. As Leonard Cohen wrote, “There is a crack in everything, that’s how the light gets in.” Life’s cracks and contradictions are not merely signs of brokenness; they are also openings through which understanding, compassion and transformation enter. """



def evaluate_cot(state: essayState):
    prompt = f"""Evaluate ONLY the clarity of thought of this essay for a UPSC Mains
aspirant: logical flow, structure, coherence between paragraphs, and quality of
introduction and conclusion. Give detailed feedback and a score out of 10.

Essay:
{state['essay']}"""
    output = str_model.invoke(prompt)
    return {'cot_feedback': output.feedback, 'cot_marks': output.score}


def evaluate_concept(state: essayState):
    prompt = f"""Evaluate ONLY the depth of content of this essay for a UPSC Mains
aspirant: conceptual understanding, relevance and variety of examples, multi-dimensional
coverage (social, political, economic, ethical) and balance of arguments. Give detailed
feedback and a score out of 10.

Essay:
{state['essay']}"""
    output = str_model.invoke(prompt)
    return {'concept_feedback': output.feedback, 'concept_marks': output.score}


def evaluate_lang(state: essayState):
    prompt = f"""Evaluate ONLY the language quality of this essay for a UPSC Mains
aspirant: vocabulary, grammar, sentence construction, tone, precision and style.
Give detailed feedback and a score out of 10.

Essay:
{state['essay']}"""
    output = str_model.invoke(prompt)
    return {'lang_feedback': output.feedback, 'lang_marks': output.score}


def final_evaluation(state: essayState):
    prompt = f"""Based on the three evaluations below, write a short, summarized overall
feedback for this UPSC Mains essay, with the key strengths and the most important
improvements.

Clarity of thought: {state['cot_feedback']}
Concept and content: {state['concept_feedback']}
Language: {state['lang_feedback']}"""
    summary = model.invoke(prompt).content      
    avg = (state['cot_marks'] + state['concept_marks'] + state['lang_marks']) / 3
    return {'final_feedback': summary, 'marks': round(avg, 1)}


graph = StateGraph(essayState)

graph.add_node('evaluate_cot', evaluate_cot)
graph.add_node('evaluate_concept', evaluate_concept)
graph.add_node('evaluate_lang', evaluate_lang)
graph.add_node('final_evaluation', final_evaluation)

# to add a conditional edge
# graph.add_conditional_edges()

# fan-out: all three start in parallel
graph.add_edge(START, 'evaluate_cot')
graph.add_edge(START, 'evaluate_concept')
graph.add_edge(START, 'evaluate_lang')

# fan-in: final_evaluation waits for all three
graph.add_edge(['evaluate_cot', 'evaluate_concept', 'evaluate_lang'], 'final_evaluation')
graph.add_edge('final_evaluation', END)

workflow = graph.compile()

result = workflow.invoke({'essay': essay_input})

print("Clarity of thought:", result['cot_marks'])
print("Concept:", result['concept_marks'])
print("Language:", result['lang_marks'])
print("Overall marks:", result['marks'])
print("\nFeedback:\n", result['final_feedback'])