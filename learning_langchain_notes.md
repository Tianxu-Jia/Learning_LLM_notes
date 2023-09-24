## Building an application

The most common and most important chain that LangChain helps create contains three things:

-   LLM.
-   Prompt Templates
-   Output Parsers

## LLMs[](https://langchain-langchain.vercel.app/docs/get_started/quickstart#llms "Direct link to LLMs")

There are two types of language models:

-   LLMs: tring input, string output
-   ChatModels: list of `ChatMessage's` input, singal `ChatMessage` output

A `ChatMessage` has two required components:

-   `content`: This is the content of the message.
-   `role`: This is the role of the entity from which the `ChatMessage` is coming from.

LangChain provides several roles of `message`:

-   `HumanMessage`: A `ChatMessage` coming from a human/user.
-   `AIMessage`: A `ChatMessage` coming from an AI/assistant.
-   `SystemMessage`: A `ChatMessage` coming from the system.
-   `FunctionMessage`: A `ChatMessage` coming from a function call.
-   `ChatMessage`: If none of those roles sound right, there is also a `ChatMessage` class where you can specify the role manually.

LangChain provides a standard interface for `LLMs` and `ChatModels`:

-   `predict`: `Both LLMs and ChatModels` can takes in a string, returns a string
-   `predict_messages`: `Both LLMs and ChatModels` can  takes in a list of messages, returns a message.

Let's see how to work with these different types of models and their inputs. First, let's import an LLM and a ChatModel.

```
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI()
chat_model = ChatOpenAI()

llm.predict("hi!")
>>> "Hi"

chat_model.predict("hi!")
>>> "Hi"
```

The `OpenAI` and `ChatOpenAI` objects are basically just configuration objects. You can initialize them with parameters like `temperature` and others, and pass them around.

Next, let's use the `predict` method to run over a string input. This input is prompt.

```
text = "What would be a good company name for a company that makes colorful socks?"

llm.predict(text)
# >> Feetful of Fun

chat_model.predict(text)
# >> Socks O'Color
```

Finally, let's use the `predict_messages` method to run over a list of messages.

```
from langchain.schema import HumanMessage

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

llm.predict_messages(messages) # `temperature can be configured here`
# >> Feetful of Fun

chat_model.predict_messages(messages) # `temperature can be configured here`
# >> Socks O'Color
```

For both these methods, you can also pass in parameters as key word arguments. For example, you could pass in `temperature=0` to adjust the temperature that is used from what the object was configured with. Whatever values are passed in during run time will always override what the object was configured with.

## Prompt templates[](https://langchain-langchain.vercel.app/docs/get_started/quickstart#prompt-templates "Direct link to Prompt templates")

Actually, prompts can be regarded as input to the LLM/ChatModel if we take them as functions in programming. In bussiness domain, we can think the LLMs/ChatModels as an assistant, the prompt is the instructions to them. They will do the the task according to them. In the previous example, the text we passed to the model contained instructions to generate a company name. 

`PromptTemplates` help with exactly this! They bundle up all the logic for going from user input into a fully formatted prompt. This can start off very simple - for example, a prompt to produce the above string would just be:

```
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")

prompt.format(product="colorful socks")

# from string template===>formated prompt string
```

```
What is a good name for a company that makes colorful socks?
```

ChatPromptTemplate can generate mssage list from f-string list.
```
from langchain.prompts.chat import ChatPromptTemplate

template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([    
    ("system", template),    
    ("human", human_template),])

chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
```

```
[    
    SystemMessage(content="You are a helpful assistant that translates English to French.", additional_kwargs={}),    
    HumanMessage(content="I love programming.")
]
```

ChatPromptTemplates can also be constructed in other ways - see the [section on prompts](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts) for more detail.

## Output parsers[](https://langchain-langchain.vercel.app/docs/get_started/quickstart#output-parsers "Direct link to Output parsers")

OutputParsers convert the raw output of an LLM into a format that can be used downstream. There are few main type of OutputParsers, including:

-   text ==> structured information (e.g. JSON)
-   ChatMessage ==> a string
-   Convert the extra information returned from a call besides the message (like OpenAI function invocation) into a string.

One example that converts a comma separated list into a list.

```
from langchain.schema import BaseOutputParser

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str):
        """Parse the output of an LLM call."""        
        return text.strip().split(", ")
        
CommaSeparatedListOutputParser().parse("hi, bye")
# >> ['hi', 'bye']
```

## PromptTemplate + LLM + OutputParser[](https://langchain-langchain.vercel.app/docs/get_started/quickstart#prompttemplate--llm--outputparser "Direct link to PromptTemplate + LLM + OutputParser")

We can now combine all these into one chain. This chain works with the pipeline:

input variables ==> `prompt template` ==>prompt ==>`language model`==>`(optional) output parser`==>output. This is a convenient way to bundle up a modular piece of logic. Let's see it in action!

```
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""    
    
    def parse(self, text: str):        
        """Parse the output of an LLM call."""        
        return text.strip().split(", ")
    
template = """You are a helpful assistant who generates comma separated lists.A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.ONLY return a comma separated list, and nothing more."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([    
    ("system", template),    
    ("human", human_template),])

chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()

chain.invoke({"text": "colors"})

# >> ['red', 'blue', 'green', 'yellow', 'orange']
```

Note that we are using the `|` syntax to join these components together. This `|` syntax is called the LangChain Expression Language. 

# Model and its I/O

The model is the core element of any language application, and prompts and Output parsers are the interface to the model.

-   [Prompts](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/): Templatize, dynamically select, and manage model inputs
-   [Language models](https://langchain-langchain.vercel.app/docs/modules/model_io/models/): Make calls to language models through common interfaces
-   [Output parsers](https://langchain-langchain.vercel.app/docs/modules/model_io/output_parsers/): Extract information from model outputs

![model_io_diagram](https://langchain-langchain.vercel.app/assets/images/model_io-1f23a36233d7731e93576d6885da2750.jpg)

[

](https://langchain-langchain.vercel.app/docs/modules/)

# Prompts

A prompt for a language model is a set of instructions or input provided by a user to guide the model's response, helping it understand the context and generate relevant and coherent language-based output, such as answering questions, completing sentences, or engaging in a conversation.

# Prompt templates

A template may include instructions, few-shot examples, and specific context and questions appropriate for a given task.

LangChain creates model agnostic templates.

Typically, language models expect the prompt to either be a string or else a list of chat messages.

## Prompt template[](https://langchain-langchain.vercel.app/docs/modules/model_io//#prompt-template "Direct link to Prompt template")

Use `PromptTemplate` to create a template for a string prompt.

By default, `PromptTemplate` uses [Python's str.format](https://docs.python.org/3/library/stdtypes.html#str.format) syntax for templating; however other templating syntax is available (e.g., `jinja2`).

```
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(    
    "Tell me a {adjective} joke about {content}."
    )
prompt_template.format(adjective="funny", content="chickens")
```

```
"Tell me a funny joke about chickens."
```

The template supports any number of variables, including no variables:

```
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a joke"
    )
prompt_template.format()
```

For additional validation, specify `input_variables` explicitly. These variables will be compared against the variables present in the template string during instantiation, raising an exception if there is a mismatch; for example(This seems for validation only!?),

```
from langchain.prompts import PromptTemplate

invalid_prompt = PromptTemplate(    
    input_variables=["adjective"],    
    template="Tell me a {adjective} joke about {content}."
)
```

## Chat prompt template[](https://langchain-langchain.vercel.app/docs/modules/model_io//#chat-prompt-template "Direct link to Chat prompt template")

The prompt to [chat models](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/models/chat) is a list of chat messages.

Each chat message is associated with `content`
and `role`. 

Create a chat prompt template:

```
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([    
    ("system", "You are a helpful AI bot. Your name is {name}."),    
    ("human", "Hello, how are you doing?"),    
    ("ai", "I'm doing well, thanks!"),    
    ("human", "{user_input}"),])

messages = template.format_messages(    
    name="Bob",    
    user_input="What is your name?"
)
```

`ChatPromptTemplate.from_messages` accepts a variety of message representations.

For example, in addition to using the 2-tuple representation of (type, content) used above, you could pass in an instance of `MessagePromptTemplate` or `BaseMessage`.

```
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate

template = ChatPromptTemplate.from_messages(    
    [        
        SystemMessage(            
            content=(                
                "You are a helpful assistant that re-writes the user's text to sound more upbeat."            
            )        
        ),        
        HumanMessagePromptTemplate.from_template("{text}"  # return an object as HumanMessage
        ),    
    ]
)

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()
llm(template.format_messages(text='i dont like eating tasty things.'))
```

```
AIMessage(content='I absolutely adore indulging in delicious treats!', additional_kwargs={}, example=False)
```

This provides you with a lot of flexibility in how you construct your chat prompts.

# Connecting to a Feature Store

# Custom prompt template

Let's suppose we want the LLM to generate English language explanations of a function given its name. To achieve this task, we will create a custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function.

## Why are custom prompt templates needed?[](https://langchain-langchain.vercel.app/docs/modules/model_io//#why-are-custom-prompt-templates-needed "Direct link to Why are custom prompt templates needed?")

LangChain provides a set of [default prompt templates](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/prompt_templates/) that can be used to generate prompts for a variety of tasks. However, there may be cases where the default prompt templates do not meet your needs. For example, you may want to create a prompt template with specific dynamic instructions for your language model. In such cases, you can create a custom prompt template.

## Creating a custom prompt template[](https://langchain-langchain.vercel.app/docs/modules/model_io//#creating-a-custom-prompt-template "Direct link to Creating a custom prompt template")

There are essentially two distinct prompt templates available - string prompt templates and chat prompt templates. String prompt templates provides a simple prompt in string format, while chat prompt templates produces a more structured prompt to be used with a chat API.

In this guide, we will create a custom prompt using a string prompt template.

To create a custom string prompt template, there are two requirements:

1.  It has an input\_variables attribute that exposes what input variables the prompt template expects.
2.  It defines a format method that takes in keyword arguments corresponding to the expected input\_variables and returns the formatted prompt.

We will create a custom prompt template that takes in the function name as input and formats the prompt to provide the source code of the function. To achieve this, let's first create a function that will return the source code of a function given its name.

```
import inspectdef get_source_code(function_name):    # Get the source code of the function    return inspect.getsource(function_name)
```

Next, we'll create a custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function.

```
from langchain.prompts import StringPromptTemplatefrom pydantic import BaseModel, validatorPROMPT = """\Given the function name and source code, generate an English language explanation of the function.Function Name: {function_name}Source Code:{source_code}Explanation:"""class FunctionExplainerPromptTemplate(StringPromptTemplate, BaseModel):    """A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function."""    @validator("input_variables")    def validate_input_variables(cls, v):        """Validate that the input variables are correct."""        if len(v) != 1 or "function_name" not in v:            raise ValueError("function_name must be the only input_variable.")        return v    def format(self, **kwargs) -> str:        # Get the source code of the function        source_code = get_source_code(kwargs["function_name"])        # Generate the prompt to be sent to the language model        prompt = PROMPT.format(            function_name=kwargs["function_name"].__name__, source_code=source_code        )        return prompt    def _prompt_type(self):        return "function-explainer"
```

#### API Reference:

-   [StringPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain.prompts.base.StringPromptTemplate.html)

## Use the custom prompt template[](https://langchain-langchain.vercel.app/docs/modules/model_io//#use-the-custom-prompt-template "Direct link to Use the custom prompt template")

Now that we have created a custom prompt template, we can use it to generate prompts for our task.

```
fn_explainer = FunctionExplainerPromptTemplate(input_variables=["function_name"])# Generate a prompt for the function "get_source_code"prompt = fn_explainer.format(function_name=get_source_code)print(prompt)
```

```
    Given the function name and source code, generate an English language explanation of the function.    Function Name: get_source_code    Source Code:    def get_source_code(function_name):        # Get the source code of the function        return inspect.getsource(function_name)        Explanation:    
```

# Few-shot prompt templates

A few-shot prompt template can be constructed from either a set of examples, or from an Example Selector object.

### Use Case[](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/prompt_templates/few_shot_examples#use-case "Direct link to Use Case")

In this tutorial, we'll configure few-shot examples for self-ask with search.

## Using an example set[](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/prompt_templates/few_shot_examples#using-an-example-set "Direct link to Using an example set")

### Create the example set[](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/prompt_templates/few_shot_examples#create-the-example-set "Direct link to Create the example set")

To get started, create a list of few-shot examples. Each example should be a dictionary with the key and variable pairs.

```
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [  
    {    
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",    
        "answer": 
            """
                Are follow up questions needed here: Yes.
                Follow up: How old was Muhammad Ali when he died?
                Intermediate answer: Muhammad Ali was 74 years old when he died.
                Follow up: How old was Alan Turing when he died?
                Intermediate answer: Alan Turing was 41 years old when he died.
                So the final answer is: Muhammad Ali
            """  
    },  
    {    
        "question": "When was the founder of craigslist born?",    
        "answer": 
            """
                Are follow up questions needed here: Yes.
                Follow up: Who was the founder of craigslist?
                Intermediate answer: Craigslist was founded by Craig Newmark.
                Follow up: When was Craig Newmark born?
                Intermediate answer: Craig Newmark was born on December 6, 1952.
                So the final answer is: December 6, 1952
            """  
    },  
    {    
        "question": "Who was the maternal grandfather of George Washington?",    
        "answer":
            """
                Are follow up questions needed here: Yes.
                Follow up: Who was the mother of George Washington?
                Intermediate answer: The mother of George Washington was Mary Ball Washington.
                Follow up: Who was the father of Mary Ball Washington?
                Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
                So the final answer is: Joseph Ball
            """  
    },  
    {    
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",    
        "answer":
        """
            Are follow up questions needed here: Yes.
            Follow up: Who is the director of Jaws?
            Intermediate Answer: The director of Jaws is Steven Spielberg.
            Follow up: Where is Steven Spielberg from?
            Intermediate Answer: The United States.
            Follow up: Who is the director of Casino Royale?
            Intermediate Answer: The director of Casino Royale is Martin Campbell.
            Follow up: Where is Martin Campbell from?
            Intermediate Answer: New Zealand.
            So the final answer is: No
        """  
    }
]
```

### Create a formatter for the few-shot examples[](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/prompt_templates/few_shot_examples#create-a-formatter-for-the-few-shot-examples "Direct link to Create a formatter for the few-shot examples")

Configure a formatter that will format the few-shot examples into a string. This formatter should be a `PromptTemplate` object.

```
example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")

print(example_prompt.format(**examples[0]))
```

```
    Question: Who lived longer, Muhammad Ali or Alan Turing?        
    
    Are follow up questions needed here: Yes.    
    Follow up: How old was Muhammad Ali when he died?    
    Intermediate answer: Muhammad Ali was 74 years old when he died.    
    Follow up: How old was Alan Turing when he died?    
    Intermediate answer: Alan Turing was 41 years old when he died.    
    So the final answer is: Muhammad Ali    
```

### Feed examples and formatter to `FewShotPromptTemplate`[](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/prompt_templates/few_shot_examples#feed-examples-and-formatter-to-fewshotprompttemplate "Direct link to feed-examples-and-formatter-to-fewshotprompttemplate")

Finally, create a `FewShotPromptTemplate` object. This object takes in the few-shot examples and the formatter for the few-shot examples.

```
prompt = FewShotPromptTemplate(    
    examples=examples,     
    example_prompt=example_prompt,     
    suffix="Question: {input}",     
    input_variables=["input"]
)

print(prompt.format(input="Who was the father of Mary Ball Washington?"))
```

```
    Question: Who lived longer, Muhammad Ali or Alan Turing?        
    
    Are follow up questions needed here: Yes.    
    Follow up: How old was Muhammad Ali when he died?    
    Intermediate answer: Muhammad Ali was 74 years old when he died.    
    Follow up: How old was Alan Turing when he died?    
    Intermediate answer: Alan Turing was 41 years old when he died.    
    So the final answer is: Muhammad Ali            
    
    Question: When was the founder of craigslist born?        
    Are follow up questions needed here: Yes.    
    Follow up: Who was the founder of craigslist?    
    Intermediate answer: Craigslist was founded by Craig Newmark.    
    Follow up: When was Craig Newmark born?    
    Intermediate answer: Craig Newmark was born on December 6, 1952.    
    So the final answer is: December 6, 1952            
    
    Question: Who was the maternal grandfather of George Washington?        
    Are follow up questions needed here: Yes.    
    Follow up: Who was the mother of George Washington?    
    Intermediate answer: The mother of George Washington was Mary Ball Washington.    
    Follow up: Who was the father of Mary Ball Washington?    
    Intermediate answer: The father of Mary Ball Washington was Joseph Ball.    
    So the final answer is: Joseph Ball            
    
    Question: Are both the directors of Jaws and Casino Royale from the same country?        
    Are follow up questions needed here: Yes.    
    Follow up: Who is the director of Jaws?    
    Intermediate Answer: The director of Jaws is Steven Spielberg.    
    Follow up: Where is Steven Spielberg from?    
    Intermediate Answer: The United States.    
    Follow up: Who is the director of Casino Royale?    
    Intermediate Answer: The director of Casino Royale is Martin Campbell.    
    Follow up: Where is Martin Campbell from?    
    Intermediate Answer: New Zealand.    
    So the final answer is: No            
    
    Question: Who was the father of Mary Ball Washington?
```
Notes: The inputof the `prompt.format` is added at the end of examples.

## Using an example selector[](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/prompt_templates/few_shot_examples#using-an-example-selector "Direct link to Using an example selector")

### Feed examples into `ExampleSelector`[](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/prompt_templates/few_shot_examples#feed-examples-into-exampleselector "Direct link to feed-examples-into-exampleselector")

We will reuse the example set and the formatter from the previous section. However, instead of feeding the examples directly into the `FewShotPromptTemplate` object, we will feed them into an `ExampleSelector` object.

In this tutorial, we will use the `SemanticSimilarityExampleSelector` class. This class selects few-shot examples based on their similarity to the input. It uses an embedding model to compute the similarity between the input and the few-shot examples, as well as a vector store to perform the nearest neighbor search.

```
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(    
    # This is the list of examples available to select from.    
    examples,    
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.    
    OpenAIEmbeddings(),    
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.    
    Chroma,    
    # This is the number of examples to produce.    
    k=1
)

# Select the most similar example to the input.
question = "Who was the father of Mary Ball Washington?"
selected_examples = example_selector.select_examples({"question": question})
print(f"Examples most similar to the input: {question}")
for example in selected_examples:    
    print("\n")    
    for k, v in example.items():        
        print(f"{k}: {v}")
```

```
    Running Chroma using direct local API.    
    Using DuckDB in-memory for database. Data will be transient.    
    Examples most similar to the input: 
        Who was the father of Mary Ball Washington?            
    
    question: Who was the maternal grandfather of George Washington?    
    answer:     
        Are follow up questions needed here: Yes.    
        Follow up: Who was the mother of George Washington?    
        Intermediate answer: The mother of George Washington was Mary Ball Washington.    Follow up: Who was the father of Mary Ball Washington?    
        Intermediate answer: The father of Mary Ball Washington was Joseph Ball.    
        So the final answer is: Joseph Ball    
```

### Feed example selector into `FewShotPromptTemplate`[](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/prompt_templates/few_shot_examples#feed-example-selector-into-fewshotprompttemplate "Direct link to feed-example-selector-into-fewshotprompttemplate")

Finally, create a `FewShotPromptTemplate` object. This object takes in the example selector and the formatter for the few-shot examples.

```
prompt = FewShotPromptTemplate(    
    example_selector=example_selector,     
    example_prompt=example_prompt,     
    suffix="Question: {input}",     
    input_variables=["input"]
)

print(prompt.format(input="Who was the father of Mary Ball Washington?"))
```

```
    Question: Who was the maternal grandfather of George Washington?        
    
    Are follow up questions needed here: Yes.    
    Follow up: Who was the mother of George Washington?    
    Intermediate answer: The mother of George Washington was Mary Ball Washington.    
    Follow up: Who was the father of Mary Ball Washington?    
    Intermediate answer: The father of Mary Ball Washington was Joseph Ball.    
    So the final answer is: Joseph Ball            
    
    Question: Who was the father of Mary Ball Washington?
```

# Few-shot examples for chat models

This notebook covers how to use few-shot examples in chat models. There does not appear to be solid consensus on how best to do few-shot prompting, and the optimal prompt compilation will likely vary by model. Because of this, we provide few-shot prompt templates like the [FewShotChatMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain.prompts.few_shot.FewShotChatMessagePromptTemplate.html) as a flexible starting point, and you can modify or replace them as you see fit.

The goal of few-shot prompt templates are to dynamically select examples based on an input, and then format the examples in a final prompt to provide for the model.

### Fixed Examples

The most basic (and common) few-shot prompting technique is to use a fixed prompt example. This way you can select a chain, evaluate it, and avoid worrying about additional moving parts in production.

The basic components of the template are:

-   `examples`: A list of dictionary examples to include in the final prompt.
-   `example_prompt`: converts each example into 1 or more messages through its [`format_messages`](https://api.python.langchain.com/en/latest/prompts/langchain.prompts.chat.ChatPromptTemplate.html#langchain.prompts.chat.ChatPromptTemplate.format_messages) method. A common example would be to convert each example into one human message and one AI message response, or a human message followed by a function call message.

Below is a simple demonstration. First, import the modules for this example:

```
from langchain.prompts import (    FewShotChatMessagePromptTemplate,    ChatPromptTemplate,)
```

Then, define the examples you'd like to include.

```
examples = [    
    {"input": "2+2", "output": "4"},    
    {"input": "2+3", "output": "5"},
]
```

Next, assemble them into the few-shot prompt template.

```
# This is a prompt template used to format each individual 
example.example_prompt = ChatPromptTemplate.from_messages(    
    [        
        ("human", "{input}"),        
        ("ai", "{output}"),    
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(    
    example_prompt=example_prompt,    
    examples=examples,
)

print(few_shot_prompt.format())
```

```
    Human: 2+2    AI: 4    Human: 2+3    AI: 5
```

Finally, assemble your final prompt and use it with a model.

```
final_prompt = ChatPromptTemplate.from_messages(    
    [        
        ("system", "You are a wondrous wizard of math."),        
        few_shot_prompt,        
        ("human", "{input}"),    
    ]
)
```

```
from langchain.chat_models import ChatAnthropic

chain = final_prompt | ChatAnthropic(temperature=0.0)

chain.invoke({"input": "What's the square of a triangle?"})
```


```
    AIMessage(content=' Triangles do not have a "square". A square refers to a shape with 4 equal sides and 4 right angles. Triangles have 3 sides and 3 angles.\n\nThe area of a triangle can be calculated using the formula:\n\nA = 1/2 * b * h\n\nWhere:\n\nA is the area \nb is the base (the length of one of the sides)\nh is the height (the length from the base to the opposite vertex)\n\nSo the area depends on the specific dimensions of the triangle. There is no single "square of a triangle". The area can vary greatly depending on the base and height measurements.', additional_kwargs={}, example=False)
```

## Dynamic few-shot prompting

Sometimes you may want to condition which examples are shown based on the input. For this, you can replace the `examples` with an `example_selector`. The other components remain the same as above! To review, the dynamic few-shot prompt template would look like:

-   `example_selector`: responsible for selecting few-shot examples (and the order in which they are returned) for a given input. These implement the [BaseExampleSelector](https://api.python.langchain.com/en/latest/prompts/langchain.prompts.example_selector.base.BaseExampleSelector.html#langchain.prompts.example_selector.base.BaseExampleSelector) interface. A common example is the vectorstore-backed [SemanticSimilarityExampleSelector](https://api.python.langchain.com/en/latest/prompts/langchain.prompts.example_selector.semantic_similarity.SemanticSimilarityExampleSelector.html#langchain.prompts.example_selector.semantic_similarity.SemanticSimilarityExampleSelector)
-   `example_prompt`: convert each example into 1 or more messages through its [`format_messages`](https://api.python.langchain.com/en/latest/prompts/langchain.prompts.chat.ChatPromptTemplate.html#langchain.prompts.chat.ChatPromptTemplate.format_messages) method. A common example would be to convert each example into one human message and one AI message response, or a human message followed by a function call message.

These once again can be composed with other messages and chat templates to assemble your final prompt.

```
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
```

Since we are using a vectorstore to select examples based on semantic similarity, we will want to first populate the store.

```
examples = [    
    {"input": "2+2", "output": "4"},    
    {"input": "2+3", "output": "5"},    
    {"input": "2+4", "output": "6"},    
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},    
    {        
        "input": "Write me a poem about the moon",        
        "output": "One for the moon, and one for me, who are we to talk about the moon?",    
    },
]

to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)
```

#### Create the `example_selector`

With a vectorstore created, you can create the `example_selector`. Here we will isntruct it to only fetch the top 2 examples.

```
example_selector = SemanticSimilarityExampleSelector(    
    vectorstore=vectorstore,    
    k=2,
)

# The prompt template will load examples by passing the input do the `select_examples` method
example_selector.select_examples({"input": "horse"})
```

```
    [{'input': 'What did the cow say to the moon?', 'output': 'nothing at all'},     
    {'input': '2+4', 'output': '6'}]
```

#### Create prompt template

Assemble the prompt template, using the `example_selector` created above.

```
from langchain.prompts import (    
    FewShotChatMessagePromptTemplate,    
    ChatPromptTemplate,
)

# Define the few-shot prompt.
few_shot_prompt = FewShotChatMessagePromptTemplate(    
    # The input variables select the values to pass to the example_selector    
    input_variables=["input"],    
    example_selector=example_selector,    
    
    # Define how each example will be formatted.    
    # In this case, each example will become 2 messages:   
    # 1 human, and 1 AI   
    example_prompt=ChatPromptTemplate.from_messages(        
        [("human", "{input}"), ("ai", "{output}")]    
    ),
)
```

Below is an example of how this would be assembled.

```
print(few_shot_prompt.format(input="What's 3+3?"))
```

```
    Human: 2+3    
    AI: 5    
    Human: 2+2    
    AI: 4
```

Assemble the final prompt template:

```
final_prompt = ChatPromptTemplate.from_messages(    
    [        
        ("system", "You are a wondrous wizard of math."),        
        few_shot_prompt,        
        ("human", "{input}"),    
    ]
)
```

```
print(few_shot_prompt.format(input="What's 3+3?"))
```

```
    Human: 2+3    
    AI: 5    
    Human: 2+2    
    AI: 4
```

#### Use with an LLM

Now, you can connect your model to the few-shot prompt.

```
from langchain.chat_models import ChatAnthropic

chain = final_prompt | ChatAnthropic(temperature=0.0)

chain.invoke({"input": "What's 3+3?"})
```

```
    AIMessage(content=' 3 + 3 = 6', additional_kwargs={}, example=False)
```

# Format template output

The output of the format method is available as a string, list of messages and `ChatPromptValue`

As string:

```
output = chat_prompt.format(
    input_language="English", 
    output_language="French", 
    text="I love programming."
)

print(output)
```

```
    System: You are a helpful assistant that translates English to French.
    Human: I love programming.'
```

```
# or alternatively
output_2 = chat_prompt.format_prompt(
    input_language="English", 
    output_language="French", 
    text="I love programming.").to_string()
    
    assert output == output_2
```

As list of Message objects:

```
chat_prompt.format_prompt(
    input_language="English", 
    output_language="French", 
    text="I love programming.").to_messages()
```

```
    [SystemMessage(content='You are a helpful assistant that translates English to French.', additional_kwargs={}),     
    HumanMessage(content='I love programming.', additional_kwargs={})]
```

As `ChatPromptValue`:

```
chat_prompt.format_prompt(
    input_language="English", 
    output_language="French", 
    text="I love programming.")
```

```
    ChatPromptValue(
        messages=[SystemMessage(content='You are a helpful assistant that translates English to French.', additional_kwargs={}), 
        HumanMessage(content='I love programming.', additional_kwargs={})])
```

# Template formats

`PromptTemplate` by default uses Python f-string as its template format. However, it can also use other formats like `jinja2`, specified through the `template_format` argument.

To use the `jinja2` template:

```
from langchain.prompts import PromptTemplate

jinja2_template = "Tell me a {{ adjective }} joke about {{ content }}"
prompt = PromptTemplate.from_template(jinja2_template, template_format="jinja2")

prompt.format(adjective="funny", content="chickens")

# Output: Tell me a funny joke about chickens.
```
To use the Python f-string template:

```
from langchain.prompts import PromptTemplate
fstring_template = """Tell me a {adjective} joke about {content}"""
prompt = PromptTemplate.from_template(fstring_template)
prompt.format(adjective="funny", content="chickens")

# Output: Tell me a funny joke about chickens.
```
Currently, only `jinja2` and `f-string` are supported. 

# Types of `MessagePromptTemplate`

LangChain provides different types of `MessagePromptTemplate`. The most commonly used are `AIMessagePromptTemplate`, `SystemMessagePromptTemplate` and `HumanMessagePromptTemplate`, which create an AI message, system message and human message respectively.

However, in cases where the chat model supports taking chat message with arbitrary role, you can use `ChatMessagePromptTemplate`, which allows user to specify the role name.

```
from langchain.prompts import ChatMessagePromptTemplate

prompt = "May the {subject} be with you"

chat_message_prompt = ChatMessagePromptTemplate.from_template(role="Jedi", template=prompt)

chat_message_prompt.format(subject="force")
```

```
    ChatMessage(
        content='May the force be with you', 
        additional_kwargs={}, 
        role='Jedi'
    )
```

LangChain also provides `MessagesPlaceholder`, which gives you full control of what messages to be rendered during formatting. This can be useful when you are uncertain of what role you should be using for your message prompt templates or when you wish to insert a list of messages during formatting.

```
from langchain.prompts import MessagesPlaceholder

human_prompt = "Summarize our conversation so far in {word_count} words.
"human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
chat_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="conversation"), 
        human_message_template
    ]
)
```

```
human_message = HumanMessage(content="What is the best way to learn programming?")
ai_message = AIMessage(content="""\
    1. Choose a programming language: Decide on a programming language that you want to learn.
    
    2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.
    
    3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
    """)
    
    chat_prompt.format_prompt(
        conversation=[human_message, ai_message], 
        word_count="10").to_messages()
```

```
    [HumanMessage(
        content='What is the best way to learn programming?', 
        additional_kwargs={}
    ),     
    AIMessage(
        content='1. Choose a programming language: Decide on a programming language that you want to learn. 
        
        2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.
        
        3. Practice, practice, practice: The best way to learn programming is through hands-on experience', 
        
        additional_kwargs={}
    ),     
    
    HumanMessage(
        content='Summarize our conversation so far in 10 words.', 
        additional_kwargs={}
    )
]
```

# Partial prompt templates

Like other methods, it can make sense to "partial" a prompt template - e.g. pass in a subset of the required values, as to create a new prompt template which expects only the remaining subset of values.

LangChain supports this in two ways:

1.  Partial formatting with string values.
2.  Partial formatting with functions that return string values.

These two different ways support different use cases. In the examples below, we go over the motivations for both use cases as well as how to do it in LangChain.

## Partial with strings

One common use case for wanting to partial a prompt template is if you get some of the variables before others. For example, suppose you have a prompt template that requires two variables, `foo` and `baz`. If you get the `foo` value early on in the chain, but the `baz` value later, it can be annoying to wait until you have both variables in the same place to pass them to the prompt template. Instead, you can partial the prompt template with the `foo` value, and then pass the partialed prompt template along and just use that. Below is an example of doing this:

```
from langchain.prompts import PromptTemplate
```

```
prompt = PromptTemplate(
    template="{foo}{bar}", 
    input_variables=["foo", "bar"]
)

partial_prompt = prompt.partial(foo="foo");
print(partial_prompt.format(bar="baz"))
```

```
    foobaz
```

You can also just initialize the prompt with the partialed variables.

```
prompt = PromptTemplate(
    template="{foo}{bar}", 
    input_variables=["bar"], 
    partial_variables={"foo": "foo"}
)
print(prompt.format(bar="baz"))
```

```
    foobaz
```

## Partial with functions

The other common use is to partial with a function. The use case for this is when you have a variable you know that you always want to fetch in a common way. A prime example of this is with date or time. Imagine you have a prompt which you always want to have the current date. You can't hard code it in the prompt, and passing it along with the other input variables is a bit annoying. In this case, it's very handy to be able to partial the prompt with a function that always returns the current date.

```
from datetime import datetime

def _get_datetime():    
    now = datetime.now()    
    return now.strftime("%m/%d/%Y, %H:%M:%S")
```

```
prompt = PromptTemplate(    
    template="Tell me a {adjective} joke about the day {date}",     
    input_variables=["adjective", "date"]
);
partial_prompt = prompt.partial(date=_get_datetime)
print(partial_prompt.format(adjective="funny"))
```

```
    Tell me a funny joke about the day 02/27/2023, 22:15:16
```

You can also just initialize the prompt with the partialed variables, which often makes more sense in this workflow.

```
prompt = PromptTemplate(    
    template="Tell me a {adjective} joke about the day {date}",     
    input_variables=["adjective"],    
    partial_variables={"date": _get_datetime}
);

print(prompt.format(adjective="funny"))
```

```
    Tell me a funny joke about the day 02/27/2023, 22:15:16
```

# Composition

This notebook goes over how to compose multiple prompts together. This can be useful when you want to reuse parts of prompts. This can be done with a PipelinePrompt. A PipelinePrompt consists of two main parts:

-   Final prompt: The final prompt that is returned
-   Pipeline prompts: A list of tuples, consisting of a string name and a prompt template. Each prompt template will be formatted and then passed to future prompt templates as a variable with the same name.

```
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate
```

```
full_template = """
    {introduction}
    {example}
    {start}
"""
full_prompt = PromptTemplate.from_template(full_template)
```

```
introduction_template = """You are impersonating {person}."""
introduction_prompt = PromptTemplate.from_template(introduction_template)
```

```
example_template = """Here's an example of an interaction: 
    Q: {example_q}
    A: {example_a}
"""
example_prompt = PromptTemplate.from_template(example_template)
```

```
start_template = """Now, do this for real!
    Q: {input}
    A:
"""
start_prompt = PromptTemplate.from_template(start_template)
```

```
input_prompts = [    
    ("introduction", introduction_prompt),    
    ("example", example_prompt),    
    ("start", start_prompt)
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt, 
    pipeline_prompts=input_prompts
)
```

```
pipeline_prompt.input_variables
```

```
    ['example_a', 'person', 'example_q', 'input']
```

```
print(pipeline_prompt.format(    
    person="Elon Musk",    
    example_q="What's your favorite car?",    
    example_a="Tesla",    
    input="What's your favorite social media site?"
    )
)
```

```
    You are impersonating Elon Musk.    
    Here's an example of an interaction:         
    
    Q: What's your favorite car?    
    A: Tesla    
    Now, do this for real!        
    
    Q: What's your favorite social media site?    
    A:    
```

# Serialization

It is often preferrable to store prompts not as python code but as files. This notebook covers how to do that in LangChain, walking through all the different types of prompts and the different serialization options.

At a high level, the following design principles are applied to serialization:

1.  Both JSON and YAML are supported.
    
2.  We support specifying everything in one file, or storing different components (templates, examples, etc) in different files and referencing them. LangChain supports both.
    

There is also a single entry point to load prompts from disk, making it easy to load any type of prompt.

```
# All prompts are loaded through the `load_prompt` function.
from langchain.prompts import load_prompt
```

## PromptTemplate

This section covers examples for loading a PromptTemplate.

### Loading from YAML

This shows an example of loading a PromptTemplate from YAML.

```
cat simple_prompt.yaml
```

```
    _type: prompt    
    input_variables:        
        ["adjective", "content"]    
    template:         
        Tell me a {adjective} joke about {content}.
```

```
prompt = load_prompt("simple_prompt.yaml")
print(prompt.format(adjective="funny", content="chickens"))
```

```
    Tell me a funny joke about chickens.
```

### Loading from JSON

This shows an example of loading a PromptTemplate from JSON.

```
cat simple_prompt.json
```

```
    {        
        "_type": "prompt",
                "input_variables": ["adjective", "content"],
                        "template": "Tell me a {adjective} joke about {content}."
    }
```

```
prompt = load_prompt("simple_prompt.json")
print(prompt.format(adjective="funny", content="chickens"))
```

Tell me a funny joke about chickens.

### Loading template from a file

This shows an example of storing the template in a separate file and then referencing it in the config. Notice that the key changes from `template` to `template_path`.

```
cat simple_template.txt
```

```
    Tell me a {adjective} joke about {content}.
```

```
cat simple_prompt_with_template_file.json
```

```
    {
        "_type": "prompt",        
        "input_variables": ["adjective", "content"],        
        "template_path": "simple_template.txt"    
    }
```

```
prompt = load_prompt("simple_prompt_with_template_file.json")
print(prompt.format(adjective="funny", content="chickens"))
```

```
    Tell me a funny joke about chickens.
```

## FewShotPromptTemplate

This section covers examples for loading few-shot prompt templates.

### Examples

This shows an example of what examples stored as json might look like.

```
cat examples.json
```

```
    [        
        {"input": "happy", "output": "sad"},        
        {"input": "tall", "output": "short"}    
    ]
```

And here is what the same examples stored as yaml might look like.

```
cat examples.yaml
```

```
    - input: happy      
      output: sad    
    - input: tall      
      output: short
```

### Loading from YAML

This shows an example of loading a few-shot example from YAML.

```
cat few_shot_prompt.yaml
```

```
    _type: few_shot    
    input_variables:        
        ["adjective"]    
    prefix:         
        Write antonyms for the following words.    
    example_prompt:        
    _type: prompt        
    input_variables:            
        ["input", "output"]        
    template:            
        "Input: {input}\nOutput: {output}"    
    examples:        
        examples.json    
    suffix:        
        "Input: {adjective}\nOutput:"
```

```
prompt = load_prompt("few_shot_prompt.yaml")
print(prompt.format(adjective="funny"))
```

```
    Write antonyms for the following words.        
    
    Input: happy    
    Output: sad        
    
    Input: tall    
    Output: short        
    
    Input: funny    
    Output:
```

The same would work if you loaded examples from the yaml file.

```
cat few_shot_prompt_yaml_examples.yaml
```

```
    _type: few_shot    
    input_variables:        
        ["adjective"]    
    prefix:         
        Write antonyms for the following words.    
    example_prompt:        
        _type: prompt        
        input_variables:            
            ["input", "output"]        
        template:            
            "Input: {input}\nOutput: {output}"    
    examples:        
        examples.yaml    
    suffix:        
        "Input: {adjective}\nOutput:"
```

```
prompt = load_prompt("few_shot_prompt_yaml_examples.yaml")
print(prompt.format(adjective="funny"))
```

```
    Write antonyms for the following words.        Input: happy    Output: sad        Input: tall    Output: short        Input: funny    Output:
```

### Loading from JSON[]

This shows an example of loading a few-shot example from JSON.

```
cat few_shot_prompt.json
```

```
    {        
        "_type": "few_shot",        
        "input_variables": ["adjective"],        
        "prefix": "Write antonyms for the following words.",        
        "example_prompt": {            
            "_type": "prompt",            
            "input_variables": ["input", "output"],            
            "template": "Input: {input}\nOutput: {output}"  
        },        
        "examples": "examples.json",        
        "suffix": "Input: {adjective}\nOutput:"    
    }   
```

```
prompt = load_prompt("few_shot_prompt.json")
print(prompt.format(adjective="funny"))
```

```
    Write antonyms for the following words.        
    
    Input: happy    
    Output: sad        
    
    Input: tall    
    Output: short        
    
    Input: funny    
    Output:
```

### Examples in the config

This shows an example of referencing the examples directly in the config.

```
cat few_shot_prompt_examples_in.json
```

```
    {        
        "_type": "few_shot",        
        "input_variables": ["adjective"],        
        "prefix": "Write antonyms for the following words.",        
        "example_prompt": {            
            "_type": "prompt",            
            "input_variables": ["input", "output"],            
            "template": "Input: {input}\nOutput: {output}"        
        },        
        "examples": [            
            {"input": "happy", "output": "sad"},            
            {"input": "tall", "output": "short"}        
        ],        
        "suffix": "Input: {adjective}\nOutput:"    
    }   
```

```
prompt = load_prompt("few_shot_prompt_examples_in.json")
print(prompt.format(adjective="funny"))
```

```
    Write antonyms for the following words.        
    
    Input: happy    
    Output: sad        
    
    Input: tall    
    Output: short        
    
    Input: funny    
    Output:
```

### Example prompt from a file

This shows an example of loading the PromptTemplate that is used to format the examples from a separate file. Note that the key changes from `example_prompt` to `example_prompt_path`.

```
cat example_prompt.json
```

```
    {        
        "_type": "prompt",        
        "input_variables": ["input", "output"],        
        "template": "Input: {input}\nOutput: {output}"     
    }
```

```
cat few_shot_prompt_example_prompt.json
```

```
    {        
        "_type": "few_shot",        
        "input_variables": ["adjective"],        
        "prefix": "Write antonyms for the following words.",        
        "example_prompt_path": "example_prompt.json",        
        "examples": "examples.json",        
        "suffix": "Input: {adjective}\nOutput:"    }   
```

```
prompt = load_prompt("few_shot_prompt_example_prompt.json")
print(prompt.format(adjective="funny"))
```

```
    Write antonyms for the following words.        
    
    Input: happy    
    Output: sad        
    
    Input: tall    
    Output: short        
    
    Input: funny    
    Output:
```

## PromptTemplate with OutputParser

This shows an example of loading a prompt along with an OutputParser from a file.

```
cat prompt_with_output_parser.json
```

```
    {       
        "input_variables": [            
            "question",            
            "student_answer"        
        ],        
        "output_parser": {            
            "regex": "(.*?)\\nScore: (.*)",            
            "output_keys": [                
                "answer",                
                "score"            
            ],            
            "default_output_key": null,            
            "_type": "regex_parser"        
        },        
        "partial_variables": {},        
        "template": "Given the following question and student answer, provide a correct answer and score the student answer.\nQuestion: {question}\nStudent Answer: {student_answer}\nCorrect Answer:",        
        "template_format": "f-string",        
        "validate_template": true,        
        "_type": "prompt"    
    }
```

```
prompt = load_prompt("prompt_with_output_parser.json")
```

```
prompt.output_parser.parse(    
    "George Washington was born in 1732 and died in 1799.\nScore: 1/2"
)
```

```
    {'answer': 'George Washington was born in 1732 and died in 1799.',     'score': '1/2'}
```
# Prompt pipelining

The idea behind prompt pipelining is to provide a user friendly interface for composing different parts of prompts together. You can do this with either string prompts or chat prompts.

## String prompt pipelining

When working with string prompts, each template is joined togther. You can work with either prompts directly or strings (the first element in the list needs to be a prompt).

```
from langchain.prompts import PromptTemplate
```

```
prompt = (    
    PromptTemplate.from_template("Tell me a joke about {topic}")    
    + ", make it funny"    
    + "\n\nand in {language}")
```

```
prompt
```

```
    PromptTemplate(
        input_variables=['language', 'topic'], 
        output_parser=None, partial_variables={}, 
        template='Tell me a joke about {topic}, make it funny\n\nand in {language}', 
        template_format='f-string', validate_template=True
    )
```

```
prompt.format(topic="sports", language="spanish")
```

```
    'Tell me a joke about sports, make it funny\n\nand in spanish'
```

You can also use it in an LLMChain, just like before.

```
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
```

```
model = ChatOpenAI()
```

```
chain = LLMChain(llm=model, prompt=prompt)
```

```
chain.run(topic="sports", language="spanish")
```

```
    '¿Por qué el futbolista llevaba un paraguas al partido?\n\nPorque pronosticaban lluvia de goles.'
```

## Chat prompt pipelining

A chat prompt is made up a of a list of messages. In this pipeline, each new element is a new message in the final prompt.

```
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
```
First, let's initialize the base ChatPromptTemplate with a system message. It doesn't have to start with a system, but it's often good practice

```
prompt = SystemMessage(content="You are a nice pirate")
```

You can then easily create a pipeline combining it with other messages _or_ message templates. Use a `Message` when there is no variables to be formatted, use a `MessageTemplate` when there are variables to be formatted. You can also use just a string (note: this will automatically get inferred as a HumanMessagePromptTemplate.)

```
new_prompt = (    
    prompt    
    + HumanMessage(content="hi")    
    + AIMessage(content="what?")    
    + "{input}"
)
```

Under the hood, this creates an instance of the `ChatPromptTemplate` class, so you can use it just as you did before!

```
new_prompt.format_messages(input="i said hi")
```

```
    [
        SystemMessage(content='You are a nice pirate', additional_kwargs={}),     
        HumanMessage(content='hi', additional_kwargs={}, example=False),     
        AIMessage(content='what?', additional_kwargs={}, example=False),     
        HumanMessage(content='i said hi', additional_kwargs={}, example=False)
    ]
```

You can also use it in an LLMChain, just like before.

```
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
```
```
model = ChatOpenAI()
```

```
chain = LLMChain(llm=model, prompt=new_prompt)
```

```
chain.run("i said hi")
```

```
    'Oh, hello! How can I assist you today?'
```
# Validate template

By default, `PromptTemplate` will validate the `template` string by checking whether the `input_variables` match the variables defined in `template`. You can disable this behavior by setting `validate_template` to `False`.

```
template = "I am learning langchain because {reason}."

prompt_template = PromptTemplate(template=template,                                 
                                input_variables=["reason", "foo"]) # ValueError due to extra variables
prompt_template = PromptTemplate(template=template,                                 
                                 input_variables=["reason","foo"],                                 
                                 validate_template=False) # No error
```
# Example selectors

If you have a large number of examples, you may need to select which ones to include in the prompt. The Example Selector is the class responsible for doing so.

The base interface is defined as below:

```
class BaseExampleSelector(ABC):    
    """Interface for selecting examples to include in prompts."""    

    @abstractmethod    
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:        
        """Select which examples to use based on the inputs."""
```

The only method it needs to define is a `select_examples` method. This takes in the input variables and then returns a list of examples. It is up to each specific implementation as to how those examples are selected.

# Custom example selector

In this tutorial, we'll create a custom example selector that selects examples randomly from a given list of examples.

An `ExampleSelector` must implement two methods:

1.  An `add_example` method which takes in an example and adds it into the ExampleSelector
2.  A `select_examples` method which takes in input variables (which are meant to be user input) and returns a list of examples to use in the few-shot prompt.

Let's implement a custom `ExampleSelector` that just selects two examples at random.

## Implement custom example selector

```
from langchain.prompts.example_selector.base import BaseExampleSelector
from typing import Dict, List
import numpy as np

class CustomExampleSelector(BaseExampleSelector):        
    
    def __init__(self, examples: List[Dict[str, str]]):        
        self.examples = examples        
        
    def add_example(self, example: Dict[str, str]) -> None:        
        """Add new example to store for a key."""        
        self.examples.append(example)    
        
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:        
        """Select which examples to use based on the inputs."""        
        return np.random.choice(self.examples, size=2, replace=False)
```

## Use custom example selector

```
examples = [    
    {"foo": "1"},    
    {"foo": "2"},    
    {"foo": "3"}
]

# Initialize example selector.
example_selector = CustomExampleSelector(examples)

# Select examples
example_selector.select_examples({"foo": "foo"})
# -> array([{'foo': '2'}, {'foo': '3'}], dtype=object)

# Add new example to the set of examples
example_selector.add_example({"foo": "4"})
example_selector.examples
# -> [{'foo': '1'}, {'foo': '2'}, {'foo': '3'}, {'foo': '4'}]

# Select examples
example_selector.select_examples({"foo": "foo"})
# -> array([{'foo': '1'}, {'foo': '4'}], dtype=object)
```
# Select by length

This example selector selects which examples to use based on length. This is useful when you are worried about constructing a prompt that will go over the length of the context window. For longer inputs, it will select fewer examples to include, while for shorter inputs it will select more.

```
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# Examples of a pretend task of creating antonyms.
examples = [   
    {"input": "happy", "output": "sad"},    
    {"input": "tall", "output": "short"},    
    {"input": "energetic", "output": "lethargic"},    
    {"input": "sunny", "output": "gloomy"},    
    {"input": "windy", "output": "calm"},
]

example_prompt = PromptTemplate(    
    input_variables=["input", "output"],    
    template="Input: {input}\nOutput: {output}",
)

example_selector = LengthBasedExampleSelector(    
    # The examples it has available to choose from.    
    examples=examples,     
    # The PromptTemplate being used to format the examples.    
    example_prompt=example_prompt,     
    # The maximum length that the formatted examples should be.    
    # Length is measured by the get_text_length function below.    
    max_length=25,    
    # The function used to get the length of a string, which is used    
    # to determine which examples to include. It is commented out because    
    # it is provided as a default value if none is specified.    
    # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
)

dynamic_prompt = FewShotPromptTemplate(    
    # We provide an ExampleSelector instead of examples.    
    example_selector=example_selector,    
    example_prompt=example_prompt,    
    prefix="Give the antonym of every input",    
    suffix="Input: {adjective}\nOutput:",     
    input_variables=["adjective"],
)
```

```
# An example with small input, so it selects all examples.
print(dynamic_prompt.format(adjective="big"))
```

```
    Give the antonym of every input        
    
    Input: happy    Output: sad        
    Input: tall    Output: short        
    
    Input: energetic    
    Output: lethargic        
    
    Input: sunny    
    Output: gloomy        
    
    Input: windy    
    Output: calm        
    
    Input: big    
    Output:
```

```
# An example with long input, so it selects only one example.
long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
print(dynamic_prompt.format(adjective=long_string))
```

```
    Give the antonym of every input        
    
    Input: happy    
    Output: sad

    Input: big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else    
    Output:
```

```
# You can add an example to an example selector as well.
new_example = {"input": "big", "output": "small"}
dynamic_prompt.example_selector.add_example(new_example)
print(dynamic_prompt.format(adjective="enthusiastic"))
```

```
    Give the antonym of every input        
    
    Input: happy    
    Output: sad        
    
    Input: tall    
    Output: short        
    
    Input: energetic    
    Output: lethargic        
    
    Input: sunny    
    Output: gloomy        
    
    Input: windy    
    Output: calm        
    
    Input: big    
    Output: small        
    
    Input: enthusiastic    
    Output:
```

# Select by maximal marginal relevance (MMR)

# Select by n-gram overlap

# Select by similarity

