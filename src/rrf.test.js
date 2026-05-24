import { test } from "node:test";
import assert from "node:assert/strict";
import { reciprocalRankFusion } from "./rrf.js";

const listFrom = (...ids) => ids.map((id) => ({ id, doc: { id } }));

test("ranks an item that appears in both lists above singletons", () => {
  const a = listFrom("a", "b", "c"); // b at rank 2
  const b = listFrom("b", "d", "a"); // b at rank 1
  const fused = reciprocalRankFusion([a, b], { k: 5, rankConstant: 60 });

  // b: 1/62 + 1/61 is the highest combined score.
  assert.equal(fused[0].id, "b");
  // a appears in both too (ranks 1 and 3) and beats the singletons d, c.
  assert.equal(fused[1].id, "a");
  assert.deepEqual(
    fused.map((d) => d.id),
    ["b", "a", "d", "c"],
  );
});

test("respects the k limit", () => {
  const fused = reciprocalRankFusion([listFrom("a", "b", "c")], { k: 2 });
  assert.equal(fused.length, 2);
});

test("attaches an rrfScore and dedupes by id", () => {
  const fused = reciprocalRankFusion([listFrom("x"), listFrom("x")], { rankConstant: 60 });
  assert.equal(fused.length, 1);
  assert.ok(Math.abs(fused[0].rrfScore - 2 / 61) < 1e-12);
});

test("handles empty input", () => {
  assert.deepEqual(reciprocalRankFusion([]), []);
  assert.deepEqual(reciprocalRankFusion([[], []]), []);
});
