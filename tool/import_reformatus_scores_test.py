import contextlib
import io
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import xml.etree.ElementTree as ET

from import_reformatus_scores import (
    _compute_song_font_size,

    _FIXED_SYLLABLE_FONT,
    _has_severe_clipping,
    _remove_header_text,
    apply_shared_score_aliases,
    compute_mask_boxes,
    compute_crop_bottom,
    detect_svg_lyric_blocks,
    extract_words_from_bbox_html,
    render_verse_svg,
    split_verse_text_to_rows,
    pick_best_search_match,
    refresh_song_scores,
    should_mask_word,
    song_numbers_for_import,
    SvgLyricBlock,
    SvgLyricRow,
)


class ComputeCropBottomTest(unittest.TestCase):
    def test_crops_before_second_verse_marker_when_present(self):
        words = [
            {'text': '90', 'y_min': 21.3, 'y_max': 48.0},
            {'text': '1.', 'y_min': 84.4, 'y_max': 98.0},
            {'text': '2.', 'y_min': 338.5, 'y_max': 352.6},
            {'text': 'Az', 'y_min': 338.5, 'y_max': 352.6},
            {'text': '3.', 'y_min': 386.5, 'y_max': 400.6},
        ]

        crop_bottom = compute_crop_bottom(page_height=501.732, words=words)

        self.assertAlmostEqual(crop_bottom, 330.5, places=1)

    def test_crops_csecsy_full_song_sheets_before_second_verse(self):
        words = [
            {'text': 'Seregeknek', 'y_min': 21.3, 'y_max': 45.1},
            {'text': '1.', 'y_min': 118.0, 'y_max': 136.0},
            {'text': '2.', 'y_min': 363.8, 'y_max': 382.3},
            {'text': 'Ím', 'y_min': 363.8, 'y_max': 382.3},
        ]

        crop_bottom = compute_crop_bottom(page_height=841.89, words=words)

        self.assertAlmostEqual(crop_bottom, 355.8, places=1)

    def test_crops_to_content_bottom_when_only_score_page_exists(self):
        words = [
            {'text': '22', 'y_min': 21.3, 'y_max': 48.0},
            {'text': '1.', 'y_min': 84.4, 'y_max': 98.0},
            {'text': 'Nagy', 'y_min': 396.0, 'y_max': 410.0},
            {'text': 'ZSOLTÁROK', 'y_min': 470.1, 'y_max': 478.9},
            {'text': '62', 'y_min': 467.6, 'y_max': 479.8},
        ]

        crop_bottom = compute_crop_bottom(page_height=501.732, words=words)

        self.assertAlmostEqual(crop_bottom, 426.0, places=1)


class ComputeMaskBoxesTest(unittest.TestCase):
    def test_scales_and_filters_words_to_cropped_first_page(self):
        words = [
            {'text': 'Aki', 'x_min': 20.0, 'y_min': 80.0, 'x_max': 60.0, 'y_max': 96.0},
            {'text': '-', 'x_min': 63.0, 'y_min': 81.0, 'x_max': 70.0, 'y_max': 95.0},
            {'text': '', 'x_min': 72.0, 'y_min': 82.0, 'x_max': 78.0, 'y_max': 95.0},
            {'text': '2.', 'x_min': 18.0, 'y_min': 330.0, 'x_max': 30.0, 'y_max': 346.0},
            {'text': 'Later', 'x_min': 20.0, 'y_min': 348.0, 'x_max': 70.0, 'y_max': 362.0},
        ]

        boxes = compute_mask_boxes(
            page_width=200.0,
            page_height=400.0,
            words=words,
            crop_bottom=340.0,
            image_width=1000,
            image_height=2000,
            padding=2,
        )

        self.assertEqual(boxes, [(90, 390, 310, 490), (305, 395, 360, 485), (80, 1640, 160, 1740)])


class ShouldMaskWordTest(unittest.TestCase):
    def test_masks_lyrics_and_punctuation_but_not_music_glyphs(self):
        self.assertTrue(should_mask_word('Aki'))
        self.assertTrue(should_mask_word('-'))
        self.assertTrue(should_mask_word('1.'))
        self.assertFalse(should_mask_word(''))
        self.assertFalse(should_mask_word(''))


class ExtractWordsFromBboxHtmlTest(unittest.TestCase):
    def test_parses_negative_coordinates_from_first_page(self):
        bbox_html = '''
        <doc>
          <page width="340.157000" height="501.732000">
            <flow>
              <block xMin="82.954701" yMin="-15.448785" xMax="87.615196" yMax="19.887983">
                <line xMin="82.954701" yMin="-15.448785" xMax="87.615196" yMax="19.887983">
                  <word xMin="82.954701" yMin="-15.448785" xMax="87.615196" yMax="19.887983"></word>
                </line>
              </block>
            </flow>
          </page>
        </doc>
        '''

        page_width, page_height, words = extract_words_from_bbox_html(bbox_html)

        self.assertEqual(page_width, 340.157)
        self.assertEqual(page_height, 501.732)
        self.assertEqual(
            words,
            [
                {
                    'text': '',
                    'x_min': 82.954701,
                    'y_min': -15.448785,
                    'x_max': 87.615196,
                    'y_max': 19.887983,
                }
            ],
        )


class PickBestSearchMatchTest(unittest.TestCase):
    def test_prefers_title_similarity_when_number_lookup_fails(self):
        candidates = [
            ('357', '/digitalis-reformatus-enekeskonyv/enek/257/', 'Buzdulj mély hálára, lelkünk'),
            ('358', '/digitalis-reformatus-enekeskonyv/enek/258/', 'Másik ének'),
        ]

        match = pick_best_search_match(
            song_number='257',
            song_title='Buzdulj mély hálára',
            candidates=candidates,
        )

        self.assertEqual(match, candidates[0])

    def test_accepts_prefix_match_when_site_title_is_longer(self):
        candidates = [
            ('343', '/digitalis-reformatus-enekeskonyv/enek/246/', 'Ó, örök Isten! Ki Atyánk vagy nékünk'),
        ]

        match = pick_best_search_match(
            song_number='255',
            song_title='Ó, örök Isten',
            candidates=candidates,
        )

        self.assertEqual(match, candidates[0])

    def test_rejects_weak_generic_search_results(self):
        candidates = [
            ('3', '/digitalis-reformatus-enekeskonyv/enek/5/', 'Ó mely sokan vannak'),
            ('5', '/digitalis-reformatus-enekeskonyv/enek/7/', 'Úr Isten, az én imádságom'),
        ]

        match = pick_best_search_match(
            song_number='318',
            song_title='Én Istenem',
            candidates=candidates,
        )

        self.assertIsNone(match)


class SongNumbersForImportTest(unittest.TestCase):
    def test_prefers_explicit_song_list(self):
        songbook = {'erdelyi': {'1': {'hasScore': True}, '2': {'hasScore': False}}}

        song_numbers = song_numbers_for_import(
            songbook,
            ['90', '481'],
            refresh_existing_scored=True,
        )

        self.assertEqual(song_numbers, ['90', '481'])

    def test_can_refresh_all_existing_scored_songs(self):
        songbook = {
            'erdelyi': {
                '1': {'hasScore': True},
                '2': {'hasScore': False},
                '3': {'hasScore': True},
            }
        }

        song_numbers = song_numbers_for_import(
            songbook,
            None,
            refresh_existing_scored=True,
        )

        self.assertEqual(song_numbers, ['1', '3'])


class SharedScoreAliasesTest(unittest.TestCase):
    def test_alias_fallback_caps_source_files_to_target_verse_count(self):
        songbook = {
            'erdelyi': {
                '89': {
                    'hasScore': True,
                    'scoreFiles': [
                        'assets/referdelyi/referdelyi-089-v001.svg',
                        'assets/referdelyi/referdelyi-089-v002.svg',
                        'assets/referdelyi/referdelyi-089-v003.svg',
                    ],
                },
                '270': {
                    'hasScore': False,
                    'texts': ['1. First verse', '2. Second verse'],
                },
                '407': {
                    'hasScore': True,
                    'scoreFiles': [
                        'assets/referdelyi/referdelyi-407-v001.svg',
                        'assets/referdelyi/referdelyi-407-v002.svg',
                    ],
                },
                '427': {'hasScore': False, 'texts': ['1. First verse']},
            }
        }

        apply_shared_score_aliases(songbook)

        self.assertEqual(
            songbook['erdelyi']['270']['scoreFiles'],
            [
                'assets/referdelyi/referdelyi-089-v001.svg',
                'assets/referdelyi/referdelyi-089-v002.svg',
            ],
        )
        self.assertEqual(
            songbook['erdelyi']['427']['scoreFiles'],
            ['assets/referdelyi/referdelyi-407-v001.svg'],
        )
        self.assertTrue(songbook['erdelyi']['427']['hasScore'])


class RefreshSongScoresTest(unittest.TestCase):
    def test_continues_after_song_level_exception(self):
        calls = []

        def importer(_songbook, song_number):
            calls.append(song_number)
            if song_number == '2':
                raise TimeoutError('temporary timeout')
            return song_number == '1'

        with contextlib.redirect_stderr(io.StringIO()):
            imported, failed = refresh_song_scores(
                {'erdelyi': {}},
                ['1', '2', '3'],
                importer=importer,
            )

        self.assertEqual(calls, ['1', '2', '3'])
        self.assertEqual(imported, ['1'])
        self.assertEqual(failed, ['2', '3'])


class DetectSvgLyricBlocksTest(unittest.TestCase):
    def test_finds_score_rows_and_footer_rows_from_svg_use_groups(self):
        svg_text = '''
        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
          <use xlink:href="#glyph-8-1" x="10" y="100" />
          <use xlink:href="#glyph-8-2" x="20" y="100" />
          <use xlink:href="#glyph-8-3" x="30" y="100" />
          <use xlink:href="#glyph-8-4" x="40" y="100" />
          <use xlink:href="#glyph-8-5" x="50" y="100" />
          <use xlink:href="#glyph-8-6" x="60" y="100" />
          <use xlink:href="#glyph-8-7" x="70" y="100" />
          <use xlink:href="#glyph-8-8" x="80" y="100" />
          <use xlink:href="#glyph-8-9" x="90" y="100" />
          <use xlink:href="#glyph-8-1" x="10" y="150" />
          <use xlink:href="#glyph-8-2" x="20" y="150" />
          <use xlink:href="#glyph-8-3" x="30" y="150" />
          <use xlink:href="#glyph-8-4" x="40" y="150" />
          <use xlink:href="#glyph-8-5" x="50" y="150" />
          <use xlink:href="#glyph-8-6" x="60" y="150" />
          <use xlink:href="#glyph-8-7" x="70" y="150" />
          <use xlink:href="#glyph-8-8" x="80" y="150" />
          <use xlink:href="#glyph-8-9" x="90" y="150" />
          <use xlink:href="#glyph-8-1" x="10" y="200" />
          <use xlink:href="#glyph-8-2" x="20" y="200" />
          <use xlink:href="#glyph-8-3" x="30" y="200" />
          <use xlink:href="#glyph-8-4" x="40" y="200" />
          <use xlink:href="#glyph-8-5" x="50" y="200" />
          <use xlink:href="#glyph-8-6" x="60" y="200" />
          <use xlink:href="#glyph-8-7" x="70" y="200" />
          <use xlink:href="#glyph-8-8" x="80" y="200" />
          <use xlink:href="#glyph-8-9" x="90" y="200" />
          <use xlink:href="#glyph-9-1" x="12" y="300" />
          <use xlink:href="#glyph-9-2" x="22" y="300" />
          <use xlink:href="#glyph-9-3" x="32" y="300" />
          <use xlink:href="#glyph-9-4" x="42" y="300" />
          <use xlink:href="#glyph-9-5" x="52" y="300" />
          <use xlink:href="#glyph-9-6" x="62" y="300" />
          <use xlink:href="#glyph-9-7" x="72" y="300" />
          <use xlink:href="#glyph-9-8" x="82" y="300" />
          <use xlink:href="#glyph-9-9" x="92" y="300" />
          <use xlink:href="#glyph-9-1" x="12" y="340" />
          <use xlink:href="#glyph-9-2" x="22" y="340" />
          <use xlink:href="#glyph-9-3" x="32" y="340" />
          <use xlink:href="#glyph-9-4" x="42" y="340" />
          <use xlink:href="#glyph-9-5" x="52" y="340" />
          <use xlink:href="#glyph-9-6" x="62" y="340" />
          <use xlink:href="#glyph-9-7" x="72" y="340" />
          <use xlink:href="#glyph-9-8" x="82" y="340" />
          <use xlink:href="#glyph-9-9" x="92" y="340" />
        </svg>
        '''

        score_block, footer_block = detect_svg_lyric_blocks(svg_text)

        self.assertEqual(score_block.font_id, '8')
        self.assertEqual([round(row.y, 1) for row in score_block.rows], [100.0, 150.0, 200.0])
        self.assertEqual(score_block.rows[0].x_min, 10.0)
        self.assertEqual(score_block.rows[0].x_max, 90.0)
        self.assertEqual(footer_block.font_id, '*')
        self.assertEqual(
            [round(row.y, 1) for row in footer_block.rows],
            [300.0, 340.0],
        )
        self.assertEqual(footer_block.rows[0].font_id, '9')

    def test_falls_back_to_dense_y_rows_when_svg_uses_do_not_have_glyph_font_ids(self):
        svg_text = '''
        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
          <use xlink:href="#A" x="10" y="100" />
          <use xlink:href="#B" x="30" y="100" />
          <use xlink:href="#C" x="50" y="100" />
          <use xlink:href="#D" x="70" y="100" />
          <use xlink:href="#A" x="10" y="150" />
          <use xlink:href="#B" x="30" y="150" />
          <use xlink:href="#C" x="50" y="150" />
          <use xlink:href="#D" x="70" y="150" />
          <use xlink:href="#A" x="10" y="200" />
          <use xlink:href="#B" x="30" y="200" />
          <use xlink:href="#C" x="50" y="200" />
          <use xlink:href="#D" x="70" y="200" />
          <use xlink:href="#title" x="5" y="20" />
        </svg>
        '''

        score_block, footer_block = detect_svg_lyric_blocks(svg_text)

        self.assertEqual(score_block.font_id, '*')
        self.assertEqual([round(row.y, 1) for row in score_block.rows], [100.0, 150.0, 200.0])
        self.assertIsNone(footer_block)


class SplitVerseTextToRowsTest(unittest.TestCase):
    def test_wraps_text_to_match_original_score_row_widths(self):
        verse_text = (
            '2. Mert ő olyan, mint a jó termőfa, '
            'Mely a víz mellett vagyon plántálva, '
            'Ő idejében meghozza gyümölcsét, '
            'És el nem szokta hullatni levelét.'
        )
        rows = split_verse_text_to_rows(
            verse_text,
            [26, 30, 28, 30],
            measure_text=lambda text: len(text),
        )

        self.assertEqual(len(rows), 4)
        self.assertTrue(rows[0].startswith('2. '))
        self.assertTrue(rows[-1].endswith('levelét.'))
        self.assertEqual(' '.join(rows), verse_text)


class WordConsistentScalingTest(unittest.TestCase):
    """Verify that syllables within the same word get a uniform font size."""

    def _build_svg(self, font_id: str, row_y: float, n_notes: int, spacing: float = 20.0) -> str:
        uses = []
        for i in range(n_notes):
            x = 50.0 + i * spacing
            uses.append(
                f'<use xlink:href="#glyph-{font_id}-{i+1}" x="{x}" y="{row_y}" />'
            )
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink">'
            + ''.join(uses)
            + '</svg>'
        )

    def _lyric_runs_by_y(self, svg_str: str) -> dict[float, list[dict]]:
        """Group the baked-lyric runs of generated-lyrics by baseline y.

        Each run is a <g> wrapping one emitted text run: a syllable, hyphen,
        or line.  Returns dicts with x (run start), sx (compress scale), and
        glyphs (referenced def ids in order).
        """
        root = ET.fromstring(svg_str)
        ns = '{http://www.w3.org/2000/svg}'
        xlink = '{http://www.w3.org/1999/xlink}'
        group = root.find(f'.//{ns}g[@id="generated-lyrics"]')
        self.assertIsNotNone(group, 'generated-lyrics group missing')
        result: dict[float, list[dict]] = {}
        for run in group:
            uses = list(run.iter(f'{ns}use'))
            if not uses:
                continue
            x, y, sx = (
                float(uses[0].get('x')), float(uses[0].get('y')), 1.0,
            )
            tf = run.find(f'{ns}g')
            if tf is not None:
                m = re.match(
                    r'translate\(([-\d.]+) ([-\d.]+)\) scale\(([\d.]+) 1\)',
                    tf.get('transform', ''),
                )
                if m:
                    x, y, sx = float(m[1]), float(m[2]), float(m[3])
            glyphs = [u.get(f'{xlink}href', '').lstrip('#') for u in uses]
            result.setdefault(round(y, 1), []).append(
                {'x': x, 'sx': sx, 'glyphs': glyphs}
            )
        return result

    def test_multi_syllable_word_has_uniform_glyph_rendering(self):
        from import_reformatus_scores import SvgLyricBlock, SvgLyricRow

        row_y = 100.0
        font_id = '8'
        # 4 notes at positions 50, 70, 90, 110 — tight gaps
        svg = self._build_svg(font_id, row_y, n_notes=4, spacing=20.0)
        row = SvgLyricRow(y=row_y, x_min=50.0, x_max=110.0, glyph_count=4)
        block = SvgLyricBlock(font_id=font_id, rows=[row])

        # "e-gek" syllabifies to ("e", True), ("gek", False)
        verse = '1. e-gek'
        onsets = [[50.0, 70.0, 90.0, 110.0]]

        result = render_verse_svg(svg, verse, block, None, onsets)
        root = ET.fromstring(result)
        ns = '{http://www.w3.org/2000/svg}'
        def_ids = {
            p.get('id') for p in root.iter(f'{ns}path') if p.get('id')
        }
        runs = sorted(
            self._lyric_runs_by_y(result)[round(row_y, 1)],
            key=lambda r: r['x'],
        )

        # '1. ', 'e', 'gek' start on the first three onsets; the hyphen
        # run may sit between 'e' and 'gek' at a midpoint position.
        onset_xs = {r['x'] for r in runs if r['glyphs'] != ['lg-hyphen']}
        for expected in (50.0, 70.0, 90.0):
            self.assertTrue(
                any(abs(x - expected) < 0.1 for x in onset_xs),
                f'no syllable run at onset {expected}; xs={onset_xs}',
            )
        # The word's syllables share the file's single glyph def set
        for run in runs:
            for gid in run['glyphs']:
                self.assertIn(gid, def_ids, f'{gid} has no path def')
        # 'gek' emits its glyphs in order at the third onset
        gek = next(r for r in runs if 'lg-k' in r['glyphs'])
        self.assertEqual(gek['glyphs'], ['lg-g', 'lg-e', 'lg-k'])
        self.assertAlmostEqual(gek['x'], 90.0, places=1)

    def test_baked_glyphs_scale_with_forced_font_size(self):
        from import_reformatus_scores import SvgLyricBlock, SvgLyricRow

        row_y = 100.0
        font_id = '8'
        svg = self._build_svg(font_id, row_y, n_notes=2, spacing=25.0)
        row = SvgLyricRow(y=row_y, x_min=50.0, x_max=75.0, glyph_count=2)
        block = SvgLyricBlock(font_id=font_id, rows=[row])

        verse = '1. nünk,'
        onsets = [[50.0, 75.0]]

        big = render_verse_svg(
            svg, verse, block, None, onsets, forced_font_size=8.0
        )
        small = render_verse_svg(
            svg, verse, block, None, onsets, forced_font_size=5.0
        )
        ns = '{http://www.w3.org/2000/svg}'

        def path_d(svg_str: str, gid: str) -> str:
            for p in ET.fromstring(svg_str).iter(f'{ns}path'):
                if p.get('id') == gid:
                    return p.get('d')
            return ''

        # 'nünk,' contains ü → udieresis; its outline must grow with font size
        d_big, d_small = path_d(big, 'lg-udieresis'), path_d(small, 'lg-udieresis')
        self.assertTrue(d_big and d_small, 'udieresis def missing')
        self.assertNotEqual(d_big, d_small, 'font size not applied to outline')
        big_coords = [float(v) for v in re.findall(r'[-\d.]+', d_big)]
        small_coords = [float(v) for v in re.findall(r'[-\d.]+', d_small)]
        self.assertAlmostEqual(
            max(big_coords) / max(small_coords), 8.0 / 5.0, places=1,
            msg='glyph outline should scale with forced font size',
        )


class HeaderStrippingTest(unittest.TestCase):
    """Verify that _remove_header_text strips page-level header glyphs."""

    _NS = '{http://www.w3.org/2000/svg}'

    def _build_svg_with_header(
        self, header_glyphs: list[tuple[float, float, str]],
        lyric_y: float, lyric_font_id: str, lyric_count: int = 4,
    ) -> str:
        """Build a minimal SVG with header text rows and a lyric row."""
        uses = []
        for y, x, fid in header_glyphs:
            uses.append(
                f'<use xlink:href="#glyph-{fid}-{int(x)}" x="{x}" y="{y}" />'
            )
        for i in range(lyric_count):
            x = 50.0 + i * 20.0
            uses.append(
                f'<use xlink:href="#glyph-{lyric_font_id}-{i+1}" x="{x}" y="{lyric_y}" />'
            )
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink">'
            + ''.join(uses)
            + '</svg>'
        )

    def test_strips_dense_header_rows_above_lyrics(self):
        """Dense glyph rows (>=8 glyphs, >15/100px) above lyrics should be removed."""
        header_glyphs = []
        for i in range(30):
            header_glyphs.append((33.839, 40.0 + i * 4.5, '3'))
        for i in range(20):
            header_glyphs.append((41.520, 40.0 + i * 5.0, '3'))

        lyric_y = 100.0
        svg = self._build_svg_with_header(header_glyphs, lyric_y, '9')
        block = SvgLyricBlock(font_id='9', rows=[
            SvgLyricRow(y=lyric_y, x_min=50.0, x_max=110.0, glyph_count=4)
        ])

        root = ET.fromstring(svg)
        _remove_header_text(root, block)

        remaining_ys = set()
        for u in root.iter(f'{self._NS}use'):
            y = u.get('y')
            if y:
                remaining_ys.add(round(float(y), 3))

        self.assertNotIn(33.839, remaining_ys, 'Title row should be stripped')
        self.assertNotIn(41.520, remaining_ys, 'Author row should be stripped')
        self.assertIn(round(lyric_y, 3), remaining_ys, 'Lyric row should remain')

    def test_preserves_sparse_musical_notation(self):
        """Sparse rows (clefs, key signatures) should NOT be stripped."""
        header_glyphs = [(65.0, 30.0, '7'), (65.0, 45.0, '7'), (65.0, 60.0, '7')]

        lyric_y = 100.0
        svg = self._build_svg_with_header(header_glyphs, lyric_y, '9')
        block = SvgLyricBlock(font_id='9', rows=[
            SvgLyricRow(y=lyric_y, x_min=50.0, x_max=110.0, glyph_count=4)
        ])

        root = ET.fromstring(svg)
        _remove_header_text(root, block)

        remaining_ys = set()
        for u in root.iter(f'{self._NS}use'):
            y = u.get('y')
            if y:
                remaining_ys.add(round(float(y), 3))

        self.assertIn(65.0, remaining_ys, 'Sparse notation row should be preserved')

    def test_no_stripping_when_no_lyrics(self):
        """Should be a no-op when score_block has no rows."""
        header_glyphs = []
        for i in range(10):
            header_glyphs.append((33.839, 40.0 + i * 5.0, '3'))
        svg = self._build_svg_with_header(header_glyphs, 100.0, '9')
        block = SvgLyricBlock(font_id='9', rows=[])

        root = ET.fromstring(svg)
        _remove_header_text(root, block)

        uses = list(root.iter(f'{self._NS}use'))
        self.assertGreater(len(uses), 0, 'No uses should be removed when no lyric rows')


class ClipPathWordBoundaryTest(unittest.TestCase):
    """Verify that clip-paths respect word boundaries."""

    def _build_svg(self, font_id: str, row_y: float, n_notes: int, spacing: float = 20.0) -> str:
        uses = []
        for i in range(n_notes):
            x = 50.0 + i * spacing
            uses.append(
                f'<use xlink:href="#glyph-{font_id}-{i+1}" x="{x}" y="{row_y}" />'
            )
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink">'
            + ''.join(uses)
            + '</svg>'
        )

    def _clip_widths(self, svg_str: str) -> dict[str, float]:
        root = ET.fromstring(svg_str)
        ns = '{http://www.w3.org/2000/svg}'
        result = {}
        for cp in root.iter(f'{ns}clipPath'):
            r = cp.find(f'{ns}rect')
            if r is not None and r.get('width'):
                result[cp.get('id', '')] = float(r.get('width'))
        return result

    def test_word_final_clip_narrower_than_within_word(self):
        """Word-final syllable clip should be narrower than within-word clip."""
        row_y = 100.0
        font_id = '8'
        svg = self._build_svg(font_id, row_y, n_notes=4, spacing=20.0)
        row = SvgLyricRow(y=row_y, x_min=50.0, x_max=110.0, glyph_count=4)
        block = SvgLyricBlock(font_id=font_id, rows=[row])

        # "e-gek az" → syllables: ("e",True), ("gek",False), ("az",False)
        verse = '1. e-gek az'
        onsets = [[50.0, 70.0, 90.0, 110.0]]

        result = render_verse_svg(svg, verse, block, None, onsets)
        clips = self._clip_widths(result)

        root = ET.fromstring(result)
        ns = '{http://www.w3.org/2000/svg}'
        syl_clips = {}
        for t in root.iter(f'{ns}text'):
            text = t.text or ''
            cp = t.get('clip-path', '')
            if cp and text not in ('-', '1. '):
                clip_id = cp.removeprefix('url(#').removesuffix(')')
                syl_clips[text] = clips.get(clip_id, 0)

        if 'e' in syl_clips and 'gek' in syl_clips:
            self.assertLess(
                syl_clips['gek'], syl_clips['e'],
                f'Word-final clip ({syl_clips["gek"]}) should be narrower '
                f'than within-word clip ({syl_clips["e"]})',
            )


class SongFontReductionTest(unittest.TestCase):
    """Verify _compute_song_font_size and _has_severe_clipping."""

    def _make_block(self, rows):
        return SvgLyricBlock(font_id='8', rows=rows)

    def _make_row(self, y, xs):
        return SvgLyricRow(
            y=y, x_min=xs[0], x_max=xs[-1],
            glyph_count=len(xs),
            glyph_xs=tuple(xs),
            glyph_ids=tuple(range(len(xs))),
        )

    def test_roomy_song_keeps_full_font(self):
        row = self._make_row(100.0, [50.0, 90.0, 130.0])
        block = self._make_block([row])
        lines = ['szent']
        onsets = [[50.0, 90.0, 130.0]]
        self.assertFalse(
            _has_severe_clipping(block.rows, lines, onsets, _FIXED_SYLLABLE_FONT)
        )
        self.assertEqual(
            _compute_song_font_size(block.rows, lines, onsets), _FIXED_SYLLABLE_FONT
        )

    def test_crowded_song_reduces_font(self):
        # Two onsets very close together; a single-char syllable cannot be
        # compressed by letter-spacing, so the font must be reduced.
        row = self._make_row(100.0, [50.0, 51.0])
        block = self._make_block([row])
        lines = ['s s']
        onsets = [[50.0, 51.0]]
        self.assertTrue(
            _has_severe_clipping(block.rows, lines, onsets, _FIXED_SYLLABLE_FONT)
        )
        self.assertLess(
            _compute_song_font_size(block.rows, lines, onsets), _FIXED_SYLLABLE_FONT
        )

    def test_no_onsets_no_reduction(self):
        row = self._make_row(100.0, [50.0, 90.0])
        block = self._make_block([row])
        lines = ['szent']
        self.assertFalse(
            _has_severe_clipping(block.rows, lines, None, _FIXED_SYLLABLE_FONT)
        )
        self.assertEqual(
            _compute_song_font_size(block.rows, lines, None), _FIXED_SYLLABLE_FONT
        )


if __name__ == '__main__':
    unittest.main()
