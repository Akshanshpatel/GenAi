from langgraph.graph import StateGraph,START,END
from typing import TypedDict

# defining state
class BmiState(TypedDict):
    weight:float
    height:float
    bmi:float
    category:str


def calculate_bmi(state:BmiState) -> BmiState:
    w=state['weight']
    h=state['height']

    bmi=(w/h**2)
    
    state['bmi']= round(bmi,2)
    return state

def category_bmi(state:BmiState)->BmiState:
    bmi=state['bmi']
    if bmi<18.5:state["category"]="Haddi"
    elif 18.5<=bmi<25: state["category"]="Fit"
    elif 25<=bmi<30: state["category"]="Mota"
    else:state["category"]="Fat fk"

    return state


# defining graph
graph=StateGraph(BmiState)

# adding nodes to graph
graph.add_node('Calculate bmi',calculate_bmi)
graph.add_node('Category bmi',category_bmi)

# adding edges to graph
graph.add_edge(START,'Calculate bmi')
graph.add_edge('Calculate bmi','Category bmi')
graph.add_edge('Category bmi',END)

# compiling and executing the graph
workflow=graph.compile()

ans=workflow.invoke({'weight':66,'height':1.72})
print(ans)