# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for data_utils.py."""

import json

from absl.testing import absltest
from absl.testing import parameterized

from exedec.data import data_utils

LLM_DATA_DEEPCODER = r'''
{"index": 0, "test_problem": {"inputs": {"x0": [[5, 8, 1], [2], [5, 2]]}, "outputs": [[5, 1], [], [5]], "program": "x0 = INPUT | x1 = Filter (%2==1) x0"}, "few_shot_examples": [{"inputs": {"x0": [[-3, 5, 0, -4], [], [2]], "x1": [5, 3, 5]}, "outputs": [[-2, 6, 1, -3], [], [3]], "program": "x0 = INPUT | x1 = INPUT | x2 = Map (+1) x0"}, {"inputs": {"x0": [[2, -4, -3], [0, -2, -2], [-6]]}, "outputs": [2, 0, -6], "program": "x0 = INPUT | x1 = Head x0"}, {"inputs": {"x0": [[1, 3], [3, 0, 1], [0, 2, 0]]}, "outputs": [[12, 36], [36, 0, 12], [0, 24, 0]], "program": "x0 = INPUT | x1 = Map (*3) x0 | x2 = Map (*4) x1"}, {"inputs": {"x0": [[0, 6, 0], [-10], [10, 5, -9]], "x1": [2, 0, 1]}, "outputs": [[0, 0, 6], [0], [-19, -9, 15]], "program": "x0 = INPUT | x1 = INPUT | x2 = Reverse x0 | x3 = Scanl1 (+) x2 | x4 = ZipWith (-) x3 x0"}]}
{"index": 1, "test_problem": {"inputs": {"x0": [[], [2, 0, 2], [0]]}, "outputs": [[], [0, 2, -2], [0]], "program": "x0 = INPUT | x1 = Scanl1 (-) x0 | x2 = ZipWith (-) x1 x0"}, "few_shot_examples": [{"inputs": {"x0": [[-13, 49], [-37, -21], [28, 34, -8, -31]], "x1": [0, 5, 0]}, "outputs": [[-1, 4], [-3, -2], [2, 3, -1, -3]], "program": "x0 = INPUT | x1 = INPUT | x2 = Map (/3) x0 | x3 = Map (+1) x2 | x4 = Map (/4) x3"}, {"inputs": {"x0": [[3, 0, 1], [0], [4, 3, 1]]}, "outputs": [[3, 3, 3], [], [4, 4, 4]], "program": "x0 = INPUT | x1 = Scanl1 (max) x0 | x2 = Filter (>0) x1"}, {"inputs": {"x0": [[-10, 8, 23], [-33], [-16, 20, -14, 38, -43]], "x1": [[1, 0, 2], [2], [0, 4]]}, "outputs": [[0, 1, -2], [3], [-1, -16]], "program": "x0 = INPUT | x1 = INPUT | x2 = ZipWith (*) x1 x1 | x3 = Map (-1) x2 | x4 = Scanl1 (-) x3"}, {"inputs": {"x0": [[-1, 6, 9, 10, 9], [7, 0, -4, 4, -5], [-7, -3]]}, "outputs": [[-1, 6, 9, 10, 9], [7, 0, -2, 4, -2], [-3, -1]], "program": "x0 = INPUT | x1 = Map (/3) x0 | x2 = ZipWith (max) x0 x1"}]}
{"index": 2, "test_problem": {"inputs": {"x0": [[2, 5], [2, 2], [7, 3, 0]], "x1": [[5], [6], [0, 8]]}, "outputs": [[6, 15], [6, 6], [-7, 9, 28]], "program": "x0 = INPUT | x1 = INPUT | x2 = Map (*4) x0 | x3 = Sort x2 | x4 = ZipWith (-) x3 x0"}, "few_shot_examples": [{"inputs": {"x0": [[-4, -12, 40, 15, 4], [21, 8, -8], []]}, "outputs": [[10, 13, 14], [5, 7], []], "program": "x0 = INPUT | x1 = Map (/4) x0 | x2 = Filter (>0) x1 | x3 = Scanl1 (+) x2"}, {"inputs": {"x0": [5, 5, 4], "x1": [[2, 2], [], [2, 1, 1, 0, 2]]}, "outputs": [[2, 2], [], [2, 1, 1, 1, 3]], "program": "x0 = INPUT | x1 = INPUT | x2 = Scanl1 (-) x1 | x3 = Scanl1 (-) x2"}, {"inputs": {"x0": [[12, -11, -41], [42, 40], [7, 37]], "x1": [[27, 27], [-48, 19], [39, 32]]}, "outputs": [[3, -3], [-12, -12], [1, 8]], "program": "x0 = INPUT | x1 = INPUT | x2 = Scanl1 (min) x1 | x3 = ZipWith (min) x0 x2 | x4 = Map (/4) x3"}, {"inputs": {"x0": [3, 1, 3], "x1": [[8, 3], [3, 4, 9], [7]]}, "outputs": [8, 3, 7], "program": "x0 = INPUT | x1 = INPUT | x2 = Head x1"}]}
'''.strip()

LLM_DATA_ROBUSTFILL = r'''
{"index": 0, "test_problem": {"inputs": [":2JwS@ ,061 @\"w{Guj@", " /tGY7:ev9G@#7 @ @:V", "@#tv ]Cl7@?wnud $626", "{J@(5D!0 @@}CD{9397 "], "outputs": [":?JwS@ ,??? @\"w{Guj@", " /tGY?:ev?G@#? @ @:V", "@#tv ]Cl?@?wnud $???", "{J@(?D!? @@}CD{???? "], "program": "16 108 84"}, "few_shot_examples": [{"inputs": ["\"\"GP#C\"!EUcN/68\"5113", "/47\"/VV&00$P\"\"0AFc\")", "\"8E{MP'1\"%292$YRAN\"}", "\",H;853,T@HIUL'FBo\"\""], "outputs": ["FFFF4\"\"GP#C\"!EUcN/68\"5113", "FFFFFFF4/47\"/VV&00$P\"\"0AFc\")", "FFF4\"8E{MP'1\"%292$YRAN\"}", "FFF4\",H;853,T@HIUL'FBo\"\""], "program": "3 16 109 50 11 105|4 75|10"}, {"inputs": ["@Uxne(Rhu.TZW VB,gj", "%OV EKW%vL,Nov'Cls", "\"HWX,sEA'WF!Oyt\"Sklz", "\"NE]Bs%AV'Kaib}Z"], "outputs": ["(R.@UXNE(RHU.TZW VB,GJ", " EKW.%OV EKW%VL,NOV'CLS", ",sEA.\"HWX,SEA'WF!OYT\"SKLZ", "]B.\"NE]BS%AV'KAIB}Z"], "program": "3 15 109 210 61 6 103 216 114 105 217 114|4 83|8 111"}, {"inputs": ["?? 7YE'kj{pkc%\"fydn%", "(yjgt)grue%?%\"vw?/U3", "&s??,zcbn)t3%!nz%", "&cxlc??$8%(yd%,kkwq"], "outputs": ["7YE'kj", "yjgt", "s", "cxlc"], "program": "3 9 97 90 6 104 216 113 107 216 114"}, {"inputs": ["#d!f:cxq(rE}2u%t&6Of", "%7vuo%Lg;c5\"arh!m;kr", ",tv7e.4UFd'zwlk/z#x]", "@ZX#o C1Ad/dp]gwqw{q"], "outputs": ["!5:5(5}5%5&5q(re}", "%5;5\"5!5;5g;c5\"", ".5'5/5#5]ufd'z", "#5 5/5]5{51ad/d"], "program": "3 16 104 76 12 104|3 8 112 5 202 206"}]}
{"index": 1, "test_problem": {"inputs": ["?DUZ$AQS]q;hFo@B'cQR", ":Z&IER$2%TKEB@Wk.zzF", "&WkFX%HZGb#FCY{DHT(L", " F,Z&SFh{MEUF{sHi)FS"], "outputs": ["'$]q;hFo@B'cQR", "'&$2%TKEB@Wk.zzF", "'%b#FCY{DHT(L", "',&SFh{MEUF{sHi)FS"], "program": "4 100|3 17 105 216 12 104"}, "few_shot_examples": [{"inputs": ["\"vdIr'ergr@ud,IBzo&r", "&te,cN?au(rrx;e", "'pqzn:hbmx$iyjA'uns(", "(wQ[xz,x[ZlaY]koY"], "outputs": ["\"zIz'z@z,IBz&z\"vd", "&z,zN z(z;z&te", "'z:z$zA'z('pqzn", "(zQ[z,z[ZzY]zY(w"], "program": "3 16 107 44 9 84 101|3 14 109 11 107"}, {"inputs": ["(!!%%%!&ymgk", "%!%;q!(%!", "%%/ieak%!(!!", "(!!%$zv%!%"], "outputs": ["%%", ";q", "ie", "%$"], "program": "3 17 109 218 5 219 221"}, {"inputs": [" VH\"urnF]79##;M#@07(", "/9#!025$aZY#g#\"3450#", "&Dep##)vu;sEYg\"108!9", "##%088:PW\"632.65#{G:"], "outputs": ["t ", "t/", "t&", "t#"], "program": "4 38|11 109"}, {"inputs": [",Wq Zt6?L%p", "\"Ba.meSr)E8%Hay", "(V)Jw.tI(GZ", ".koX\"J;UE%kpaK"], "outputs": [",", "\"", "(", "."], "program": "11 109"}]}
{"index": 2, "test_problem": {"inputs": [")lpnz&rb3.wfo'MC/FIJ", "(KN/cpl.J?JUu#el", "}F]cs&glj U]t", "!ITVq)U!gfcm,XFTR$ks"], "outputs": [")&.'/b/", "(/.?#b?JUu#", "}]& ]b]cs&glj U]", "!)!,$b!gfcm,XFTR$"], "program": "18 104|4 20|6 105 213 114 104 214 113"}, "few_shot_examples": [{"inputs": [":;snvf]:(to]/)Tpws", " zCao]?Shk}k/:]:", "]: Xcuh]:/.Ye[G", "?L]]/ Eam(QZc::"], "outputs": ["to]/)Tvf]:(to", "k}k/:]o]?N}", "/.Ye[Gcuh]:/.", "(QZc::/ N(Q"], "program": "3 14 109 5 225 230|3 16 106 58 5 220 226"}, {"inputs": [",,$CM.N&Dtp,%Bqa)Tww", ",@EZB,}Jpcx}LGZ]Pjfz", ",OZ,.D[BD(Bjnz[Vt,#S", "}Y,[Yery.OOIB,\"A,;Qg"], "outputs": [",,$CM.N&\"tp,%Bqa)Tww", ",@EZB,}Jpcx}\"]Pjfz", ",OZ,.D[\"(Bjnz[Vt,#S", "}Y,[Yery.\",\"A,;Qg"], "program": "3 14 109 15 105 218 99"}, {"inputs": ["}awBn,hn((RUd$ckg(", "!XxOD('p{X($9r", ")Paz(,mJR]UJ(]g", "(]b?g?m(.zK"], "outputs": ["}P", "!P", ")P", "(P"], "program": "3 16 103 86 11 109|4 60"}, {"inputs": ["[vh:@Hrj Lrxb Su}bhx", ":@rmcw,cxee{bdyr\"Oam", ":Aq@Gcld::tkk@Gg%v}d", "(Ozfp$qqrv:/Goit]rkd"], "outputs": ["Hrj0d", "Oam0d", "Aq0d", "Ozfp0d"], "program": "7 106 216|4 71|4 22"}]}
'''.strip()

TEST_DATA_DEEPCODER = r'''
{"index": 0, "inputs": ["x0 = [ ] | x1 = [ 0 ]", "x0 = [ 1 0 6 9 1 ] | x1 = [ 9 ]", "x0 = [ 3 7 1 4 ] | x1 = [ -3 -1 ]"], "outputs": ["[ ]", "[ 5 14 15 ]", "[ 4 5 9 ]"], "program": "x0 = INPUT | x1 = INPUT | x2 = Scanl1 (-) x0 | x3 = Map (*(-1)) x2 | x4 = Filter (>0) x3"}
{"index": 1, "inputs": ["x0 = [ ] | x1 = [ 0 ]", "x0 = [ 1 0 6 9 1 ] | x1 = [ 9 ]", "x0 = [ 3 7 1 4 ] | x1 = [ -3 -1 ]"], "outputs": ["[ ]", "[ 1 0 18 27 3 ]", "[ -3 -3 1 7 ]"], "program": "x0 = INPUT | x1 = INPUT | x2 = Map (*3) x0 | x3 = Scanl1 (min) x0 | x4 = ZipWith (*) x0 x3 | x5 = ZipWith (+) x3 x4 | x6 = ZipWith (-) x2 x5"}
{"index": 2, "inputs": ["x0 = [ ] | x1 = [ 0 ]", "x0 = [ 1 0 6 9 1 ] | x1 = [ 9 ]", "x0 = [ 3 7 1 4 ] | x1 = [ -3 -1 ]"], "outputs": ["[ ]", "[ 1 -3 2 6 1 ]", "[ 1 5 -1 2 ]"], "program": "x0 = INPUT | x1 = INPUT | x2 = Map (/3) x0 | x3 = ZipWith (-) x0 x2 | x4 = Reverse x2 | x5 = ZipWith (-) x3 x4"}
'''.strip()

TEST_DATA_ROBUSTFILL = r'''
{"index": 0, "inputs": ["#My##:Gxbo[Ned[Er%", "#%$Ua.Qaeq?Opa%Kcr#", "%{Eos#(Mdjt#'Yi{Oclf", "%##Tq@Fh#Xza#?Fdlu"], "outputs": ["k[MY##:GXBO[NED[ER%8y##:Gxbo[Ned[", "kK%$UA.QAEQ?OPA%KCR#8aUa.Qaeq?Opa%", "kO{EOS#(MDJT#'YI{OCLF8osos#(Mdjt#'Yi", "kF##TQ@FH#XZA#?FDLU8qTq@Fh#Xza#?F"], "program": "4 29|7 109 211|3 8 111 17 109 216|3 15 109 216 79 7 106 216|5 219 230"}
{"index": 1, "inputs": ["$\"jd%MzCO{bP[Iu?z$", "$.dvp\"2{O86F!vh$5$", "!e5Kp&8z/M02Z$$[q}ee", ".qL\"q$[hnlc o3Z#lAt$"], "outputs": ["d%MzCO{bP[Iu?$\"jd%MzCO{bP[Iu?z$$\"%MCO{P[I?$$\"%MzCO{bP[Iu?z$\"&$", "vp\"2{O86F!$.dvp\"2{O86F!vh$5$$.\"2{O86F!$5$$.\"2{O86F!vh$5$\"&$", "Kp&8z/M02Z$$[q}!e5Kp5Kp&8z/M02Z$$[q}ee!5K&8/M02Z$$[}!5Kp&8z/M02Z$$[q}ee\"&e", "\"q$[hnlc o3Z#lA.qLL\"q$[hnlc o3Z#lAt$.L\"$[ 3Z#A$.L\"q$[hnlc o3Z#lAt$\"&$"], "program": "6 109 219 113 107 214 113|11 104|12 107|18 107|17 107 216|4 99|4 81|7 109 214"}
{"index": 2, "inputs": ["?wm:i))%86[hGuN)", "?ga))%6:f$GEci)", "@QIvE]jHQ{241)))&sWy", ")(C#7573%Ai)%vv)"], "outputs": ["?wm:i))%8wmihGuNY8?wm:}))%86[hGuN)?wm:i))%86[hGuN)h?wm:i))%86[hGuN)?wm:i))%86[hG'N)?w", "?ga))%6gafGEciY6?ga))%6:}$GEci)?ga))%6:f$GEci)fGE?ga))%6:f$GEci)?ga))%6:f$GE'i)?g", "@QIvE]jHQ{2QIvEjHQsWyY2@QIvE]}{241)))&sWy@QIvE]jHQ{241)))&sWyQ@QIvE]jHQ{241)))&sWy@QIvE]jHQ{241)))&'Wy@Q", ")(C#7CAivvY7)(C#7573%})%vv))(C#7573%Ai)%vv)Ai)(C!7573%Ai)%vv))(C#7573%Ai)%'v))("], "program": "11 108|13 103 218|4 69|3 13 108 216 6 103 212 113 103 214 114|3 10 15 103 213 93|10|3 14 103 5 224 227|9 98 85|15 109 212 100|13 109 217"}
'''.strip()


class DataUtilsTest(parameterized.TestCase):

  def test_dsl_program_to_python_deepcoder(self):
    dsl_program = 'x0 = INPUT | x1 = Map (+1) x0 | x2 = Scanl1 (+) x1'
    self.assertEqual(
        data_utils.dsl_program_to_python(
            dsl_program, 'deepcoder', pythonic=False),
        '''
def program(x0):
  x1 = dsl.Map(dsl.PLUS_ONE, x0)
  x2 = dsl.Scanl1(dsl.ADD, x1)
  return x2
'''.strip())
    self.assertEqual(
        data_utils.dsl_program_to_python(
            dsl_program, 'deepcoder', pythonic=True),
        '''
def program(x0):
  x1 = [x + 1 for x in x0]
  x2 = dsl.Scanl1(lambda x, y: x + y, x1)
  return x2
'''.strip())

  def test_dsl_program_to_python_robustfill(self):
    dsl_program = '5 216 218|4 83|3 8 111 7 103 217'
    self.assertEqual(
        data_utils.dsl_program_to_python(dsl_program, 'robustfill'),
        '''
def program(x):
  parts = [
      dsl.SubStr(x, 1, 3),
      dsl.Const('.'),
      dsl.ToCase(dsl.GetToken(x, dsl.Type.WORD, 2), dsl.Case.ALL_CAPS),
  ]
  return ''.join(parts)
'''.strip())

  def test_run_python_program_deepcoder(self):
    dsl_program = 'x0 = INPUT | x1 = Map (+1) x0 | x2 = Scanl1 (+) x1'
    for pythonic in [True, False]:
      python_program = data_utils.dsl_program_to_python(
          dsl_program, 'deepcoder', pythonic=pythonic)
      outputs = data_utils.run_python_program(
          python_program,
          inputs={'x0': [[2, 3, 4], [7, -2], [], 4]},
          dataset_type='deepcoder')
      self.assertEqual(outputs, [[3, 7, 12], [8, 7], [], None])

  def test_run_python_program_robustfill(self):
    dsl_program = '5 216 218|4 83|3 8 111 7 103 217'
    python_program = data_utils.dsl_program_to_python(dsl_program, 'robustfill')
    outputs = data_utils.run_python_program(
        python_program,
        inputs=['apple (banana) clementine', 'durian:elderberry', '', 8],
        dataset_type='robustfill')
    self.assertEqual(outputs, ['app.BANANA', 'dur.ELDERBERRY', '.', None])

  @parameterized.named_parameters(
      ('deepcoder', 'deepcoder', False),
      ('deepcoder_pythonic', 'deepcoder', True),
      ('robustfill', 'robustfill', False),
  )
  def test_llm_data(self, dataset_type, pythonic):
    jsonl_content = (LLM_DATA_DEEPCODER if dataset_type == 'deepcoder'
                     else LLM_DATA_ROBUSTFILL)
    data = [json.loads(line) for line in jsonl_content.splitlines()]
    for d in data:
      self.assertLen(d['few_shot_examples'], 4)
      problems = [d['test_problem']] + d['few_shot_examples']
      for p in problems:
        python_program = data_utils.dsl_program_to_python(
            p['program'], dataset_type, pythonic)
        outputs = data_utils.run_python_program(
            python_program, p['inputs'], dataset_type)
        self.assertEqual(outputs, p['outputs'])

  @parameterized.named_parameters(
      ('deepcoder', 'deepcoder', 'x0 = [ 3 7 1 4 ] | x1 = [ -3 -1 ]'),
      ('robustfill', 'robustfill', 'Apple!'),
  )
  def test_spec_conversions(self, dataset_type, spec_str):
    spec_id_to_token, spec_token_to_id = data_utils.spec_vocab_tables(
        dataset_type)
    spec_ids = data_utils.spec_str_to_ids(spec_str, dataset_type,
                                          spec_token_to_id)
    self.assertEqual(data_utils.spec_ids_to_str(spec_ids, dataset_type,
                                                spec_id_to_token),
                     spec_str)

  @parameterized.named_parameters(
      ('deepcoder', 'deepcoder'),
      ('robustfill', 'robustfill'),
  )
  def test_test_data(self, dataset_type):
    jsonl_content = (TEST_DATA_DEEPCODER if dataset_type == 'deepcoder'
                     else TEST_DATA_ROBUSTFILL)
    data = [json.loads(line) for line in jsonl_content.splitlines()]
    program_id_to_token, program_token_to_id = data_utils.program_vocab_tables(
        dataset_type)
    for d in data:
      program_ids = data_utils.program_str_to_ids(
          d['program'], dataset_type, program_token_to_id)
      program = data_utils.ids_to_program(
          program_ids, dataset_type, program_id_to_token)
      outputs = data_utils.run_program(program, d['inputs'], dataset_type)
      self.assertEqual(outputs, d['outputs'])


if __name__ == '__main__':
  absltest.main()
