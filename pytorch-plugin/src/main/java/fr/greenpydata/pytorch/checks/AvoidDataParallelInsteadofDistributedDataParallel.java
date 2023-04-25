package fr.greenpydata.pytorch.checks;

import java.util.Optional;

import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.CallExpression;

@Rule(
  key = AvoidDataParallelInsteadofDistributedDataParallel.RULE_KEY,
  name = AvoidDataParallelInsteadofDistributedDataParallel.MESSAGE,
  description = AvoidDataParallelInsteadofDistributedDataParallel.MESSAGE,
  priority = Priority.MINOR,
  tags = {"bug", "eco-design"})
public class AvoidDataParallelInsteadofDistributedDataParallel extends PythonSubscriptionCheck {

  public static final String RULE_KEY = "P1";

  private static final String dataParallelFullyQualifiedName = "torch.nn.DataParallel";

  protected static final String MESSAGE = "Use DistributedDataParallel instead of DataParallel.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol symbol = callExpression.calleeSymbol();
      Optional.ofNullable(symbol)
        .map(Symbol::fullyQualifiedName)
        .filter(e -> e.equals(dataParallelFullyQualifiedName))
        .ifPresent(functionFqn -> ctx.addIssue(callExpression, MESSAGE));
    });
  }

}
