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

# Small model (generates feedback)
small_lm = dspy.LM(
    model='ollama/qwen2.5:0.5b',
    api_base='http://localhost:11434',
    api_key='',
    temperature=0.7
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
    """Generate brutally honest, actionable feedback on code or ideas."""

    content = dspy.InputField(desc="Code snippet, idea, or text to review")
    context = dspy.InputField(desc="Additional context about what the user wants")
    feedback = dspy.OutputField(desc="Brutally honest, specific, actionable feedback")


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
        "gold_feedback": "This is competent but verbose. The manual loop is unnecessary when sum() exists. You're reinventing the wheel. Replace the entire function body with 'return sum(numbers)'. Also, the function name is redundant - 'sum' alone would be fine unless you're avoiding the builtin for some reason."
    },
    {
        "content": "We should build a social network for developers where they can share code snippets and get feedback.",
        "context": "Evaluate this startup idea",
        "gold_feedback": "This has been done to death. GitHub, Stack Overflow, CodePen, Replit, and dozens of others already exist. What's your differentiator? 'Social network for developers' is not a unique value proposition. You need to identify a specific pain point that existing solutions don't address. Right now, this is just 'me too' thinking."
    },
    {
        "content": "I think we should refactor this module to use dependency injection so it's more testable.",
        "context": "Is this a good idea?",
        "gold_feedback": "Maybe, but you're jumping to a solution before identifying the problem. Why is it currently hard to test? Is it because of tight coupling, external dependencies, or something else? Dependency injection adds complexity - is the testing benefit worth it? Show me the test you can't write now, and we'll see if DI is actually the right answer or if you're just cargo-culting a pattern."
    },
    {
        "content": "class User:\n    def __init__(self, name, email, age, address, phone):\n        self.name = name\n        self.email = email\n        self.age = age\n        self.address = address\n        self.phone = phone",
        "context": "Review this class design",
        "gold_feedback": "This will become unmaintainable fast. Five constructor parameters is already pushing it. When you add the next field, you'll break every instantiation. Use a builder pattern, use keyword arguments, or better yet, use a dataclass. Also, you're missing validation - what happens if someone passes age=-5 or email='not an email'? This is naive and will cause bugs."
    },
    {
        "content": "for i in range(len(my_list)):\n    print(my_list[i])",
        "context": "How can I improve this?",
        "gold_feedback": "You're writing C in Python. Stop using range(len()). This is a code smell that screams 'I don't understand Python idioms'. Use 'for item in my_list' or if you need indices, 'for i, item in enumerate(my_list)'. The code you wrote is both harder to read and slower."
    },
    {
        "content": "Let's make our API backwards compatible by supporting both the old and new formats indefinitely.",
        "context": "Is this smart?",
        "gold_feedback": "No. This is how you end up with technical debt that crushes you. Backwards compatibility is important, but 'indefinitely' is insane. You'll be maintaining two code paths forever, testing both, and confusing new developers. Set a deprecation timeline. Support the old format for 6-12 months with loud warnings, then kill it. Document the migration path clearly. Rip the band-aid off or you'll die by a thousand cuts."
    },
    {
        "content": "try:\n    result = risky_operation()\nexcept:\n    pass",
        "context": "Review this error handling",
        "gold_feedback": "This is unacceptable. You're silencing ALL exceptions, including KeyboardInterrupt and SystemExit. This will make debugging impossible. You're hiding failures that could crash your program in mysterious ways later. At minimum, catch specific exceptions. Better yet, log the error. Best, let it fail fast and fix the root cause. Silent failures are bugs waiting to happen."
    },
    {
        "content": "We should use microservices because they're more scalable.",
        "context": "Good architecture choice?",
        "gold_feedback": "Cargo cult engineering. Microservices don't automatically mean scalable - they mean distributed systems complexity. You'll deal with network failures, distributed transactions, service discovery, and deployment orchestration. Can you not scale a monolith? Have you actually hit resource limits? Most companies running microservices would be better off with a well-designed monolith. Don't add complexity without proving you need it."
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
        "content": "# TODO: Fix this later\nresult = hacky_workaround()",
        "context": "Review this code comment",
    },
    {
        "content": "Let's rewrite the entire codebase in Rust for better performance.",
        "context": "Should we do this?",
    },
]


# ==============================================================================
# MODULES
# ==============================================================================

class BrutalFeedbackModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(FeedbackGenerator)

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
