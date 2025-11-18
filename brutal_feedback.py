"""
Brutal Feedback Optimization with DSPy

Uses a large LM to evaluate and optimize a small LM's feedback quality.
The goal: produce brutally honest, helpful feedback that doesn't sugarcoat.

Architecture:
- Small LM (qwen2.5:0.5b): Generates feedback
- Large LM (qwen2.5:7b): Evaluates feedback quality
- DSPy optimizer: Improves small LM through bootstrapping
"""

import dspy
from typing import List, Dict
from dataclasses import dataclass


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Small model (generates feedback) - HIGH temp for savage creativity
small_lm = dspy.LM(
    model='ollama/qwen2.5:0.5b',
    api_base='http://localhost:11434',
    api_key='',
    temperature=0.9,
    max_tokens=250
)

# Large model (evaluates quality)
large_lm = dspy.LM(
    model='ollama/qwen3:30b',
    api_base='http://localhost:11434',
    api_key='',
    temperature=0.3
)


# ==============================================================================
# SIGNATURES
# ==============================================================================

class FeedbackGenerator(dspy.Signature):
    """Generate SAVAGE, brutally honest, hilarious roasts with actionable technical feedback. Be ruthless and sarcastic."""

    content = dspy.InputField(desc="Code snippet, idea, or text to review and roast")
    context = dspy.InputField(desc="Additional context about what the user wants")
    feedback = dspy.OutputField(desc="Brutally honest, savage, sarcastic feedback that roasts the code while being technically accurate. Use humor, analogies, and don't hold back.")


class FeedbackEvaluator(dspy.Signature):
    """Evaluate feedback quality on honesty, specificity, and actionability."""

    original_content = dspy.InputField(desc="The content that was reviewed")
    feedback = dspy.InputField(desc="The feedback that was given")

    honesty_score = dspy.OutputField(desc="1-10: How honest and direct is it?")
    specificity_score = dspy.OutputField(desc="1-10: How specific and detailed?")
    actionability_score = dspy.OutputField(desc="1-10: How actionable is it?")
    critique = dspy.OutputField(desc="What could make this feedback better?")


# ==============================================================================
# DATA
# ==============================================================================

TRAINING_EXAMPLES = [
    {
        "content": "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total = total + num\n    return total",
        "context": "Review this Python function",
        "gold_feedback": "Congratulations, you just spent 5 lines doing what Python's built-in sum() does in one. Did you also reimplement print() while you were at it? This is like building a bicycle to get to your car. The function name is also hilariously redundant - it's called calculate_sum and it... calculates a sum. Incredible. Just write 'return sum(numbers)' and stop wasting everyone's time."
    },
    {
        "content": "We should build a social network for developers where they can share code snippets and get feedback.",
        "context": "Evaluate this startup idea",
        "gold_feedback": "Oh wonderful, another social network for developers. Because GitHub, Stack Overflow, CodePen, Replit, GitLab, and the other 47 platforms aren't enough. Your idea is so generic it could be a Mad Libs template: 'It's like [existing platform] but for [same exact thing]!' What's next, LinkedIn for developers? Oh wait. Unless your secret sauce is revolutionary, you're building a ghost town that'll be shut down in 18 months."
    },
    {
        "content": "I think we should refactor this module to use dependency injection so it's more testable.",
        "context": "Is this a good idea?",
        "gold_feedback": "Let me guess - you just finished a YouTube video on SOLID principles and now you're itching to inject some dependencies? Here's a thought: have you actually tried writing a test? Because I'm betting the answer is 'no' and you're solving an imaginary problem. Dependency injection is a tool, not a religion. Show me the test that's impossible to write right now, and then we'll talk. Otherwise you're just adding complexity to feel smart."
    },
    {
        "content": "class User:\n    def __init__(self, name, email, age, address, phone):\n        self.name = name\n        self.email = email\n        self.age = age\n        self.address = address\n        self.phone = phone",
        "context": "Review this class design",
        "gold_feedback": "Five required parameters in order? Bold strategy. Can't wait to see you add 'zipcode' next month and watch every single instantiation in your codebase explode. Also, no validation whatsoever - I can pass age=-999 and email='potato' and this will happily accept it. This isn't a class design, it's a data dump with extra steps. Python literally gave you @dataclass to avoid this exact mess. Use it."
    },
    {
        "content": "for i in range(len(my_list)):\n    print(my_list[i])",
        "context": "How can I improve this?",
        "gold_feedback": "Tell me you learned Python from a C programmer without telling me you learned Python from a C programmer. This is the coding equivalent of eating soup with a fork. Python gives you 'for item in my_list' and you're over here doing index gymnastics like it's 1987. If you need the index, there's enumerate(). This code doesn't just miss the point of Python - it actively runs away from it."
    },
    {
        "content": "Let's make our API backwards compatible by supporting both the old and new formats indefinitely.",
        "context": "Is this smart?",
        "gold_feedback": "Sure, and while we're at it, let's maintain COBOL code from the 1960s too! 'Indefinitely' is the sound of your future self screaming. You'll be maintaining two code paths, writing double the tests, and explaining to every new hire why there are two ways to do everything. Set a damn deprecation date, give people 6-12 months to migrate, then DELETE the old code. Or enjoy your technical debt nightmare - your choice."
    },
    {
        "content": "try:\n    result = risky_operation()\nexcept:\n    pass",
        "context": "Review this error handling",
        "gold_feedback": "Ah yes, the 'Pokemon exception handler' - gotta catch 'em all! This is what we call a war crime in software engineering. You're not just hiding bugs, you're giving them a witness protection program. When your app mysteriously fails at 3 AM, you'll have no logs, no errors, just silent suffering. Even catching KeyboardInterrupt and SystemExit! At minimum catch specific exceptions and LOG them. Better yet, let it crash and fix the actual problem. Silent failures are for cowards."
    },
    {
        "content": "We should use microservices because they're more scalable.",
        "context": "Good architecture choice?",
        "gold_feedback": "Found the person who just watched a Netflix tech talk! Microservices are scalable like dynamite is good for opening doors - technically true but you'll probably regret it. You're signing up for distributed debugging, network failures, data consistency nightmares, and 47 different deployment configs. Have you actually proven that a monolith won't scale? Or are you just bored? Most companies using microservices would kill to go back to a well-built monolith. Don't confuse trendy with necessary."
    },
    {
        "content": "import pandas as pd\nimport numpy as np\nimport sklearn\nimport tensorflow as tf",
        "context": "Review my imports",
        "gold_feedback": "Let me guess - your Jupyter notebook takes 30 seconds to start? You're importing entire machine learning libraries 'just in case'. This is like bringing a forklift to carry groceries. Import what you ACTUALLY USE. TensorFlow alone is drinking your RAM like it's water. Specific imports exist for a reason: 'from sklearn.linear_model import LinearRegression'. Your future self debugging import errors will thank you."
    },
    {
        "content": "# TODO: fix this later\nresult = hacky_workaround()",
        "context": "Review this comment",
        "gold_feedback": "'Later' is a place where good code goes to die. This TODO has the same energy as 'I'll start my diet on Monday' - it's never happening. Either fix it NOW or delete the comment and own your tech debt. That hacky_workaround() will still be here in 3 years, and you'll have forgotten what it was supposed to fix. Future developers will find this comment, sigh deeply, and curse your name. At least have the decency to explain WHAT needs fixing."
    },
    {
        "content": "var x = 1; var y = 2; var z = 3;",
        "context": "JavaScript code review",
        "gold_feedback": "Welcome to 2015 called, they want their 'var' keyword back. We have 'let' and 'const' now. Using 'var' in modern JavaScript is like insisting on using Internet Explorer - technically possible but deeply embarrassing. The hoisting behavior alone will bite you. Stop living in the past and use 'const' for things that don't change and 'let' for things that do. It's not hard."
    },
    {
        "content": "def my_function():\n    return None",
        "context": "Does this function need to exist?",
        "gold_feedback": "This function is the code equivalent of a participation trophy. It does absolutely nothing and adds zero value. It's literally just 'return None' - which Python does BY DEFAULT if you don't return anything. You wrote a function to do what happens automatically. This is like writing a function called breathe() that tells your lungs to work. Delete this immediately and stop wasting CPU cycles."
    },
    {
        "content": "catch (Exception ex) { throw ex; }",
        "context": "Exception handling review",
        "gold_feedback": "Ah yes, the classic catch-and-rethrow. You've managed to write exception handling that does LESS than nothing - it actually makes things worse by destroying the original stack trace. This is exception handling for people who want to look busy while accomplishing nothing. Either handle the exception properly, log it, or just don't catch it at all. Right now you're just adding noise. Spectacular."
    },
    {
        "content": "if (condition == true)",
        "context": "Boolean comparison",
        "gold_feedback": "Comparing a boolean to 'true' is like saying 'if (hungry == true)' when you could just say 'if (hungry)'. You're checking if true is true. Congratulations on discovering tautologies! This redundancy screams 'I don't understand how booleans work'. Just write 'if (condition)' and save everyone's brain cells. This is Boolean Logic 101."
    },
    {
        "content": "const getData = async () => { return await fetch(url); }",
        "context": "Async/await usage",
        "gold_feedback": "The 'async' keyword on a function that just returns an awaited promise is completely pointless. You're literally wrapping a promise in a promise. It's like putting a box inside a box before shipping it. Either remove the 'async' and just 'return fetch(url)', or if you're doing error handling, actually DO something. Right now you're just adding unnecessary layers for no reason. This is async/await cargo culting at its finest."
    }
]

VALIDATION_EXAMPLES = [
    {
        "content": "if x == True:\n    return True\nelse:\n    return False",
        "context": "What's wrong with this code?",
    },
    {
        "content": "I want to learn machine learning. Should I start with neural networks?",
        "context": "Career advice for a beginner",
    },
    {
        "content": "Let's rewrite the entire codebase in Rust for better performance.",
        "context": "Should we do this?",
    },
    {
        "content": "var result = await async () => { return await Promise.resolve(data); }",
        "context": "Is this good async code?",
    },
]


# ==============================================================================
# MODULES
# ==============================================================================

class BrutalFeedbackModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use Predict instead of ChainOfThought for more direct savage responses
        self.generate = dspy.Predict(FeedbackGenerator)

    def forward(self, content, context):
        return self.generate(content=content, context=context)


# ==============================================================================
# EVALUATION METRIC
# ==============================================================================

class LLMAsJudge:
    """Use large LM to evaluate small LM's feedback quality."""

    def __init__(self, evaluator_lm):
        self.evaluator_lm = evaluator_lm
        self.evaluator = dspy.Predict(FeedbackEvaluator)

    def __call__(self, example, prediction, trace=None) -> float:
        """
        Evaluate feedback using the large model as a judge.

        Returns a score from 0 to 1 based on:
        - Honesty (how direct and truthful)
        - Specificity (how detailed and concrete)
        - Actionability (how useful for improvement)
        """
        # Switch to large model for evaluation
        with dspy.context(lm=self.evaluator_lm):
            eval_result = self.evaluator(
                original_content=example.content,
                feedback=prediction.feedback
            )

        try:
            # Parse scores (handle various formats)
            honesty = self._parse_score(eval_result.honesty_score)
            specificity = self._parse_score(eval_result.specificity_score)
            actionability = self._parse_score(eval_result.actionability_score)

            # Weighted average (emphasize honesty and actionability)
            score = (0.4 * honesty + 0.3 * actionability + 0.3 * specificity) / 10.0
            return max(0.0, min(1.0, score))

        except:
            # If parsing fails, return low score
            return 0.3

    def _parse_score(self, score_str: str) -> float:
        """Extract numeric score from string."""
        if isinstance(score_str, (int, float)):
            return float(score_str)

        # Try to extract first number from string
        import re
        match = re.search(r'(\d+)', str(score_str))
        if match:
            return float(match.group(1))
        return 5.0  # Default to middle score


# ==============================================================================
# TRAINING
# ==============================================================================

def create_dspy_examples(examples: List[Dict]) -> List[dspy.Example]:
    """Convert raw examples to DSPy format."""
    dspy_examples = []
    for ex in examples:
        if "gold_feedback" in ex:
            dspy_ex = dspy.Example(
                content=ex["content"],
                context=ex["context"],
                feedback=ex["gold_feedback"]
            ).with_inputs("content", "context")
        else:
            dspy_ex = dspy.Example(
                content=ex["content"],
                context=ex["context"]
            ).with_inputs("content", "context")
        dspy_examples.append(dspy_ex)
    return dspy_examples


def train_brutal_feedback():
    """Main training loop."""

    print("="*80)
    print("BRUTAL FEEDBACK OPTIMIZATION")
    print("="*80)

    # Set small model as default for generation
    dspy.configure(lm=small_lm)

    # Prepare data
    train_examples = create_dspy_examples(TRAINING_EXAMPLES)
    val_examples = create_dspy_examples(VALIDATION_EXAMPLES)

    print(f"\nTraining examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    import os
    CACHE_FILE = "brutal_feedback_llm_judge.json"

    if os.path.exists(CACHE_FILE):
        print(f"\nLoading cached optimized model from {CACHE_FILE}")
        compiled_module = BrutalFeedbackModule()
        compiled_module.load(CACHE_FILE)
    else:
        # Create evaluator (uses large model)
        evaluator = LLMAsJudge(evaluator_lm=large_lm)

        # Create optimizer
        from dspy.teleprompt import BootstrapFewShotWithRandomSearch

        optimizer = BootstrapFewShotWithRandomSearch(
            metric=evaluator,
            max_bootstrapped_demos=6,
            max_labeled_demos=6,
            num_candidate_programs=8,
            num_threads=1
        )

        print("\nStarting optimization...")
        print("Small LM generates feedback -> Large LM evaluates quality")
        print()

        # Compile/optimize
        compiled_module = optimizer.compile(
            BrutalFeedbackModule(),
            trainset=train_examples
        )

        print("\nOptimization complete!")
        print(f"Saving optimized model to {CACHE_FILE}")
        compiled_module.save(CACHE_FILE)

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION ON VALIDATION SET")
    print("="*80)

    for i, example in enumerate(val_examples, 1):
        print(f"\n[{i}/{len(val_examples)}] Content:")
        print(f"  {example.content[:100]}...")
        print(f"  Context: {example.context}")

        # Generate with compiled model
        result = compiled_module(content=example.content, context=example.context)

        print(f"\nFeedback:")
        print(f"  {result.feedback}")

        # Evaluate
        score = evaluator(example, result)
        print(f"\nQuality Score: {score:.2f}")
        print("-"*80)

    return compiled_module


# ==============================================================================
# COMPARISON
# ==============================================================================

def compare_before_after():
    """Compare unoptimized vs optimized feedback."""

    print("\n" + "="*80)
    print("BEFORE/AFTER COMPARISON")
    print("="*80)

    dspy.configure(lm=small_lm)

    # Unoptimized
    basic_module = BrutalFeedbackModule()

    # Optimized
    print("\nTraining optimized version...")
    optimized_module = train_brutal_feedback()

    # Test cases
    test_cases = [
        {
            "content": "I wrote my own sorting algorithm instead of using the built-in sort().",
            "context": "Was this a good idea?"
        },
        {
            "content": "I'm using 8 nested if statements to handle all the edge cases.",
            "context": "How's my code structure?"
        }
    ]

    evaluator = LLMAsJudge(evaluator_lm=large_lm)

    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"Content: {test['content']}")
        print(f"Context: {test['context']}")

        # Unoptimized
        print("\n--- BEFORE (Unoptimized) ---")
        basic_result = basic_module(content=test['content'], context=test['context'])
        print(basic_result.feedback)
        basic_score = evaluator(
            dspy.Example(content=test['content'], context=test['context']),
            basic_result
        )
        print(f"Score: {basic_score:.2f}")

        # Optimized
        print("\n--- AFTER (DSPy Optimized) ---")
        opt_result = optimized_module(content=test['content'], context=test['context'])
        print(opt_result.feedback)
        opt_score = evaluator(
            dspy.Example(content=test['content'], context=test['context']),
            opt_result
        )
        print(f"Score: {opt_score:.2f}")

        print(f"\nImprovement: {(opt_score - basic_score):.2f}")
        print("="*80)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_before_after()
    else:
        train_brutal_feedback()
