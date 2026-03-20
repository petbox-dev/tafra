// Syntax highlighting enhancements for source code blocks
document.addEventListener("DOMContentLoaded", function () {
  // Known type names to highlight regardless of context
  var TYPE_NAMES = new Set([
    // tafra types
    "Tafra", "InitVar", "ObjectFormatter", "GroupSet",
    "Union", "GroupBy", "Transform", "IterateBy",
    "InnerJoin", "LeftJoin", "CrossJoin",
    "GroupDescription", "InitAggregation",
    // typing / collections.abc
    "Any", "Callable", "Iterator", "Iterable", "Mapping",
    "Sequence", "Sized", "Optional", "MutableMapping",
    "KeysView", "ValuesView", "ItemsView",
    "Protocol", "Literal", "ParamSpec", "Concatenate",
    // numpy
    "ndarray", "dtype",
    // Python stdlib
    "Path", "TextIOWrapper", "IO",
    "DataFrame", "Series", "Cursor",
    "NamedTuple",
  ]);

  document.querySelectorAll(".highlight code").forEach(function (block) {
    var spans = block.querySelectorAll("span.n");

    for (var i = 0; i < spans.length; i++) {
      var span = spans[i];
      var next = span.nextElementSibling;
      var prev = span.previousElementSibling;
      var text = span.textContent;

      // 1. Function/method calls: span.n followed by span.p starting with "("
      if (next && next.classList.contains("p") && next.textContent.charAt(0) === "(") {
        span.classList.add("fn-call");
        continue;
      }

      // 2. Known type names — always teal
      if (TYPE_NAMES.has(text)) {
        span.classList.add("type-name");
        continue;
      }

      // 3. Return type annotation: span.n after "->" operator
      if (prev && prev.classList.contains("o") && prev.textContent.trim() === "->") {
        span.classList.add("type-name");
        continue;
      }

      // 4. Parameter type annotation: span.n after ":" BUT only in
      //    function signatures (inside def ... parens), not after if/for/while
      //    Detect by checking: prev is ":", and the span before ":" is span.n
      //    (parameter name), and we're inside a def signature
      if (prev && prev.classList.contains("p") && prev.textContent.trim() === ":") {
        var beforeColon = prev.previousElementSibling;
        // Must be after a parameter name (span.n) or closing bracket (span.p "]")
        // and NOT after a keyword like if/for/while/else
        if (beforeColon && (beforeColon.classList.contains("n") ||
            beforeColon.classList.contains("bp") ||
            (beforeColon.classList.contains("p") && beforeColon.textContent.indexOf("]") !== -1))) {
          // Walk backwards to verify we're inside a def signature (find "def" keyword)
          var inDef = false;
          var parenDepth = 0;
          var walker = prev;
          while (walker) {
            var wt = walker.textContent;
            if (wt.indexOf(")") !== -1) parenDepth++;
            if (wt.indexOf("(") !== -1) parenDepth--;
            if (parenDepth < 0) {
              // We exited parens — check if preceding element is a def
              var defCheck = walker.previousElementSibling;
              while (defCheck && defCheck.classList.contains("w")) {
                defCheck = defCheck.previousElementSibling;
              }
              if (defCheck && defCheck.classList.contains("k") && defCheck.textContent.trim() === "def") {
                inDef = true;
              }
              break;
            }
            walker = walker.previousElementSibling;
          }
          if (inDef) {
            span.classList.add("type-name");
            continue;
          }
        }
      }

      // 5. Type inside brackets: span.n after "[" or after "," inside type brackets
      //    Only if the bracket context started from a known type or annotation
      if (prev && prev.classList.contains("p")) {
        var pt = prev.textContent.trim();
        if (pt === "[" || pt === ",") {
          // Walk back to find the opening context
          var w = prev;
          var d = 0;
          var isTypeContext = false;
          while (w) {
            var c = w.textContent;
            if (c.indexOf("]") !== -1) d++;
            if (c.indexOf("[") !== -1) d--;
            if (d < 0) {
              // Found the outermost "[" — check what's before it
              var before = w.previousElementSibling;
              if (before && (before.classList.contains("type-name") ||
                  before.classList.contains("nb") ||
                  before.classList.contains("ne") ||
                  TYPE_NAMES.has(before.textContent))) {
                isTypeContext = true;
              }
              break;
            }
            w = w.previousElementSibling;
          }
          if (isTypeContext) {
            span.classList.add("type-name");
            continue;
          }
        }
      }
    }
  });
});
