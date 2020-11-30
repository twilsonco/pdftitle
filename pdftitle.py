# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name
import sys
import argparse
import traceback
import os
import string
import re
from io import StringIO
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.psparser import literal_name
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfinterp import PDFInterpreterError
from pdfminer.pdfdevice import PDFDevice
from pdfminer import utils
from pdfminer.pdffont import PDFUnicodeNotDefined
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

VERBOSE = False
MISSING_CHAR = None
WITHIN_WORD_MOVE_LIMIT = 0
ALGO = "original"
TITLE_CASE = False


def set_globals(verbose=False, missing_char=None, within_word_move_limit=0, algo='original', title_case=False):
    global VERBOSE, MISSING_CHAR, WITHIN_WORD_MOVE_LIMIT, ALGO, TITLE_CASE
    VERBOSE = verbose
    MISSING_CHAR = missing_char
    WITHIN_WORD_MOVE_LIMIT = within_word_move_limit
    ALGO = algo
    TITLE_CASE = title_case

def verbose(*s):
    if VERBOSE:
        print(*s)


def verbose_operator(*s):
    if VERBOSE:
        print(*s)


class TextState():
    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        # charspace added to each glyph after rendering
        # this is not the width of glyph, this is extra, so default is 0
        # operator Tc
        # unscaled text space units
        self.Tc = 0
        # similar to charspace but applies only to space char=ascii 32
        # operator Tw
        # unscaled text space units
        self.Tw = 0
        # applies always horizontally
        # scales individual glyph widths by this
        # that is why default (scale of operator Tz) is 100, 100%, no change
        # operator Tz
        self.Th = 1
        # distance between the baselines of adjacent text lines
        # always applies to vertical coordinate
        # operator TL
        # unscaled text space units
        self.Tl = 0
        # operator Tf selects both font and font size
        self.Tf = None
        self.Tfs = None
        # only about rendering
        # operator Tr
        self.Tmode = 0
        # moves baseline up or down, so setting this to 0 resets it
        # operator Ts
        # unscaled text space units
        self.Trise = 0
        # text matrix
        self.Tm = None
        # text line matrix
        self.Tlm = None

    def __repr__(self):
        return ('<TextState: f=%r, fs=%r, c=%r, w=%r, '
                'h=%r, l=%r, mode=%r, rise=%r, '
                'm=%r, lm=%r>' %
                (self.Tf, self.Tfs, self.Tc, self.Tw,
                 self.Th, self.Tl, self.Tmode, self.Trise,
                 self.Tm, self.Tlm))

    def on_BT(self):
        self.Tm = utils.MATRIX_IDENTITY
        self.Tlm = utils.MATRIX_IDENTITY

    def on_ET(self):
        self.Tm = None
        self.Tlm = None


class TextOnlyInterpreter(PDFPageInterpreter):
    # pylint: disable=too-many-public-methods

    def __init__(self, rsrcmgr, device):
        PDFPageInterpreter.__init__(self, rsrcmgr, device)
        self.mpts = TextState()

    # omit these operators
    def do_w(self, linewidth):
        pass

    def do_J(self, linecap):
        pass

    def do_j(self, linejoin):
        pass

    def do_M(self, miterlimit):
        pass

    def do_d(self, dash, phase):
        pass

    def do_ri(self, intent):
        pass

    def do_i(self, flatness):
        pass

    def do_m(self, x, y):
        pass

    def do_l(self, x, y):
        pass

    def do_c(self, x1, y1, x2, y2, x3, y3):  # pylint: disable=too-many-arguments
        pass

    def do_y(self, x1, y1, x3, y3):
        pass

    def do_h(self):
        pass

    def do_re(self, x, y, w, h):
        pass

    def do_S(self):
        pass

    def do_s(self):
        pass

    def do_f(self):
        pass

    def do_f_a(self):
        pass

    def do_B(self):
        pass

    def do_B_a(self):
        pass

    def do_b(self):
        pass

    def do_b_a(self):
        pass

    def do_n(self):
        pass

    def do_W(self):
        pass

    def do_W_a(self):
        pass

    def do_CS(self, name):
        pass

    def do_cs(self, name):
        pass

    def do_G(self, gray):
        pass

    def do_g(self, gray):
        pass

    def do_RG(self, r, g, b):
        pass

    def do_rg(self, r, g, b):
        pass

    def do_K(self, c, m, y, k):
        pass

    def do_k(self, c, m, y, k):
        pass

    def do_SCN(self):
        pass

    def do_scn(self):
        pass

    def so_SC(self):
        pass

    def do_sc(self):
        pass

    def do_sh(self, name):
        pass

    def do_EI(self, obj):
        pass

    def do_Do(self, xobjid):
        pass

    # text object begin/end
    def do_BT(self):
        verbose_operator("PDF OPERATOR BT")
        self.mpts.on_BT()

    def do_ET(self):
        verbose_operator("PDF OPERATOR ET")
        self.mpts.on_ET()

    # text state operators
    def do_Tc(self, space):
        verbose_operator("PDF OPERATOR Tc: space=", space)
        self.mpts.Tc = space

    def do_Tw(self, space):
        verbose_operator("PDF OPERATOR Tw: space=", space)
        self.mpts.Tw = space

    def do_Tz(self, scale):
        verbose_operator("PDF OPERATOR Tz: scale=", scale)
        self.mpts.Th = scale * 0.01

    def do_TL(self, leading):
        verbose_operator("PDF OPERATOR TL: leading=", leading)
        self.mpts.Tl = leading

    def do_Tf(self, fontid, fontsize):
        verbose_operator("PDF OPERATOR Tf: fontid=", fontid,
                         ", fontsize=", fontsize)
        try:
            self.mpts.Tf = self.fontmap[literal_name(fontid)]
            verbose_operator("font=", self.mpts.Tf.fontname)
            self.mpts.Tfs = fontsize
        except KeyError:
            # pylint: disable=raise-missing-from
            raise PDFInterpreterError('Undefined Font id: %r' % fontid)

    def do_Tr(self, render):
        verbose_operator("PDF OPERATOR Tr: render=", render)
        self.mpts.Tmode = render

    def do_Ts(self, rise):
        verbose_operator("PDF OPERATOR Ts: rise=", rise)
        self.mpts.Trise = rise

    # text-move operators

    def do_Td(self, tx, ty):
        verbose_operator("PDF OPERATOR Td: tx=", tx, ", ty=", ty)
        self.mpts.Tlm = utils.translate_matrix(self.mpts.Tlm, (tx, ty))
        self.mpts.Tm = self.mpts.Tlm

    def do_TD(self, tx, ty):
        verbose_operator("PDF OPERATOR TD: tx=", tx, ", ty=", ty)
        self.do_TL(-ty)
        self.do_Td(tx, ty)

    def do_Tm(self, a, b, c, d, e, f):  # pylint: disable=too-many-arguments
        verbose_operator("PDF OPERATOR Tm: matrix=", (a, b, c, d, e, f))
        self.mpts.Tlm = (a, b, c, d, e, f)
        self.mpts.Tm = self.mpts.Tlm

    # T*
    def do_T_a(self):
        verbose_operator("PDF OPERATOR T*")
        self.do_Td(0, self.mpts.Tl)

    # text-showing operators

    # show a string
    def do_Tj(self, s):
        verbose_operator("PDF operator Tj: s=", s)
        self.do_TJ([s])

    # ' quote
    # move to next line and show the string
    # same as:
    # T*
    # string Tj
    def do__q(self, s):
        verbose_operator("PDF operator q: s=", s)
        self.do_T_a()
        self.do_Tj(s)

    # " doublequote
    # move to next line and show the string
    # using aw word spacing, ac char spacing
    # same as:
    # aw Tw
    # ac Tc
    # string '
    def do__w(self, aw, ac, s):
        verbose_operator("PDF OPERATOR \": aw=", aw,
                         ", ac=", ac, ", s=", s)
        self.do_Tw(aw)
        self.do_Tc(ac)
        self.do__q(s)

    # show one or more text string, allowing individual glyph positioning
    # each element in the array is either a string or a number
    # if string, it is the string to show
    # if number, it is the number to adjust text position, it translates Tm
    def do_TJ(self, seq):
        verbose_operator("PDF OPERATOR TJ: seq=", seq)
        self.device.process_string(self.mpts, seq)


class TextOnlyDevice(PDFDevice):

    def __init__(self, rsrcmgr):
        PDFDevice.__init__(self, rsrcmgr)
        self.last_state = None
        # contains (font, font_size, string)
        self.blocks = []
        # current block
        # font, font size, glyph y, [chars]
        self.current_block = None

    # at the end of the file, we need to recover last paragraph
    def recover_last_paragraph(self):
        if len(self.current_block[4]) > 0:
            self.blocks.append(self.current_block)

    # pdf spec, page 410
    def new_tx(self, w, Tj, Tfs, Tc, Tw, Th):  # pylint: disable=no-self-use,too-many-arguments
        return ((w - Tj / 1000) * Tfs + Tc + Tw) * Th

    # pdf spec, page 410
    def new_ty(self, w, Tj, Tfs, Tc, Tw):  # pylint: disable=no-self-use,too-many-arguments
        return (w - Tj / 1000) * Tfs + Tc + Tw

    def process_string(self, ts, array):
        verbose('SHOW STRING ts: ', ts)
        verbose('SHOW STRING array: ', array)
        for obj in array:
            verbose("processing obj: ", obj)
            # this comes from TJ, number translates Tm
            if utils.isnumber(obj):
                Tj = obj
                verbose("processing translation: ", Tj)
                # translating Tm, change tx, ty according to direction
                if ts.Tf.is_vertical():
                    tx = 0
                    ty = self.new_ty(0, Tj, ts.Tfs, 0, ts.Tw)
                else:
                    tx = self.new_tx(0, Tj, ts.Tfs, 0, ts.Tw, ts.Th)
                    ty = 0
                # update Tm accordingly
                ts.Tm = utils.translate_matrix(ts.Tm, (tx, ty))
                # there is an heuristic needed here, not sure what
                # if -Tj > ts.Tf.char_width('o'):
                #    self.draw_cid(ts, 0, force_space=True)
            else:
                verbose("processing string")
                for cid in ts.Tf.decode(obj):
                    self.draw_cid(ts, cid)

    # pylint: disable=too-many-branches
    def draw_cid(self, ts, cid, force_space=False):
        verbose("drawing cid: ", cid)
        Trm = utils.mult_matrix((ts.Tfs * ts.Th, 0, 0, ts.Tfs, 0, ts.Trise),
                                ts.Tm)
        if Trm[1] != 0:
            return
        if Trm[2] != 0:
            return
        verbose('Trm', Trm)
        if cid == 32 or force_space:
            Tw = ts.Tw
        else:
            Tw = 0
        try:
            if force_space:
                unichar = ' '
            else:
                try:
                    unichar = ts.Tf.to_unichr(cid)
                except Exception as e:
                    verbose(f"Failed to process {cid = }: {e}")
                    unichar = ' '
        except PDFUnicodeNotDefined:
            if MISSING_CHAR:
                unichar = MISSING_CHAR
            else:
                raise
        (gx, gy) = utils.apply_matrix_pt(Trm, (0, 0))
        verbose("drawing unichar: '", unichar, "' @", gx, ",", gy)
        tfs = Trm[0]
        if self.current_block is None:
            self.current_block = (ts.Tf, tfs, gx, gy, [unichar])
        elif ((self.current_block[0] == ts.Tf) and
              (self.current_block[1] == tfs)):
            self.current_block[4].append(unichar)
        else:
            self.blocks.append(self.current_block)
            self.current_block = (ts.Tf, tfs, gx, gy, [unichar])
        verbose('current block: ', self.current_block)
        verbose('blocks: ', self.blocks)
        if force_space:
            pass
        else:
            w = ts.Tf.char_width(cid)
            if ts.Tf.is_vertical():
                tx = 0
                ty = self.new_ty(w, 0, ts.Tfs, ts.Tc, Tw)
            else:
                tx = self.new_tx(w, 0, ts.Tfs, ts.Tc, Tw, ts.Th)
                ty = 0
            ts.Tm = utils.translate_matrix(ts.Tm, (tx, ty))


# pylint: disable=too-many-branches, too-many-locals, too-many-statements
def get_title_from_io(pdf_io, min_ch, min_wd):
    parser = PDFParser(pdf_io)
    # if pdf is protected with a pwd, 2nd param here is password
    doc = PDFDocument(parser)

    # pdf may not allow extraction
    # pylint: disable=no-else-return
    if doc.is_extractable:
        rm = PDFResourceManager()
        dev = TextOnlyDevice(rm)
        interpreter = TextOnlyInterpreter(rm, dev)

        first_page = StringIO()
        converter = TextConverter(rm, first_page, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(rm, converter)

        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
            page_interpreter.process_page(page)
            break

        converter.close()
        first_page_text = first_page.getvalue()
        first_page.close()
        dev.recover_last_paragraph()
        verbose('all blocks')

        for b in dev.blocks:
            verbose(b)
        
        title = None
        max_tfs_cutoff = None
        tfs_tol = 1
        y_tol = 1
        max_num_iter = 4 # number of times to lower max_tfs_cutoff if title too short or too few words.
        tfs_iter = 0
        while tfs_iter < max_num_iter and (not title or (min_ch > 1 and len(title) < min_ch) or (min_wd > 1 and len(title.split(' ')) > 1 and len(title.split(' ')) < min_wd)):
            tfs_iter += 1
            # pylint: disable=W0603
            # global ALGO # don't neet 'global ALGO' as it's not being modified, can still read it.
            if ALGO == "original":
                # find max font size
                max_tfs = max([b for b in dev.blocks if (not max_tfs_cutoff or b[1] < max_tfs_cutoff)], key=lambda x: x[1])[1]
                verbose('max_tfs: ', max_tfs)
                # find max blocks with max font size
                max_blocks = list(filter(lambda x: abs(x[1] - max_tfs) < tfs_tol, dev.blocks))
                # find the one with the highest y coordinate
                # this is the most close to top
                max_y = max(max_blocks, key=lambda x: x[3])[3]
                verbose('max_y: ', max_y)
                found_blocks = list(filter(lambda x: abs(x[3] - max_y) < y_tol, max_blocks))
                verbose('found blocks')

                for b in found_blocks:
                    verbose(b)
                    
                title = ''
                for b in found_blocks:
                    title += ''.join(b[4])
            elif ALGO == "max2":
                # find max font size
                all_tfs = sorted(list(map(lambda x: x[1], [b for b in dev.blocks if (not max_tfs_cutoff or b[1] < max_tfs_cutoff)])), reverse=True)
                max_tfs = all_tfs[0]
                verbose('max_tfs: ', max_tfs)
                selected_blocks = []
                max2_tfs = -1
                for b in dev.blocks:
                    if max2_tfs == -1:
                        if abs(b[1] - max_tfs) < tfs_tol:
                            selected_blocks.append(b)
                        elif len(selected_blocks) > 0: # max is added
                            selected_blocks.append(b)
                            max2_tfs = b[1]
                    else:
                        if abs(b[1] - max_tfs) < tfs_tol or abs(b[1] - max2_tfs) < tfs_tol:
                            selected_blocks.append(b)
                        else:
                            break

                for b in selected_blocks:
                    verbose(b)

                title = ''
                for b in selected_blocks:
                    title += ''.join(b[4])
            elif ALGO == "max_position":
                # find max font size
                max_tfs = max([b for b in dev.blocks if (not max_tfs_cutoff or b[1] < max_tfs_cutoff)], key=lambda x: x[1])[1]
                verbose('max_tfs: ', max_tfs)
                # find max blocks with max font size
                tfs_tol = 1
                max_blocks = [b for b in dev.blocks if abs(b[1] - max_tfs) < tfs_tol]
                for b in max_blocks:
                    verbose(b)
                # Now use the y-range of max_blocks as the check
                # for all blocks, with a much higher tolerance for
                # tfs to account for sub/superscript characters which
                # can vary by +/- 10pts.
                y_max = max(max_blocks, key=lambda x: x[3])[3]
                y_min = min(max_blocks, key=lambda x: x[3])[3]
                y_range = y_max - y_min
                y_mid = (y_max + y_min) * 0.5
                verbose(f"{y_range = }, {y_mid = }")
                # find the one with the highest y coordinate
                # this is the most close to top
                y_tol = 2
                tfs_tol = 8
                found_blocks = [b for b in dev.blocks if b in max_blocks or (b[3] <= y_max + y_tol and b[3] >= y_min - y_tol and abs(b[1] - max_tfs) < tfs_tol)]
                verbose('found blocks')

                for b in found_blocks:
                    verbose(b)
                    
                title = ''
                for b in found_blocks:
                    title += ''.join(b[4])
            else:
                raise Exception("unsupported ALGO")
            
        
            max_tfs_cutoff = max_tfs
            
            verbose(f"before retrieving spaces, {title = }")
            
            # Retrieve missing spaces if needed
            # if " " not in title:
            #     title = retrieve_spaces(first_page_text, title)
            new_title = retrieve_spaces_word_based(first_page_text, title.replace(' ',''))
            if len(new_title) > len(title):
                title = new_title

            # Remove duplcate spaces if any are present
            if "  " in title:
                title = " ".join(title.split())

        return title
    else:
        return None


def get_title_from_file(pdf_file, min_ch = 5, min_wd = 0):
    with open(pdf_file, 'rb') as raw_file:
        return get_title_from_io(raw_file, min_ch, min_wd)


def retrieve_spaces(first_page, title_without_space):
    # Correct the space problem
    #  if the document does not use space character between the words
    # Stop condition : all the first page has been explored or
    #  we have explored all the letters of the title

    p=0
    t=0
    result=""
    
    verbose(first_page)
    
    while p < len(first_page) and t < len(title_without_space):
        verbose(f"comparing {first_page[p] = } to {title_without_space[t] = }")
        if first_page[p].lower() == title_without_space[t].lower():
            result += first_page[p]
            verbose(f"Added {first_page[p] = } to {result = }")
            t += 1
        elif t != 0:
            # Add spaces if there is space or a wordwrap
            if first_page[p] == " " or first_page[p] == "\n":
                result += " "
            # If letter p-1 in page corresponds to letter t-1 in title,
            #  but letter p does not corresponds to letter t,
            # we are not exploring the title in the page
            else:
                t = 0
                result = ""
                
        p += 1
    
    return result

def retrieve_spaces_word_based(first_page, title_without_space):
    # for this approach, search for individual words
    # using str.find, which searches from the left,
    # so when a word is found it's likely the first
    # word of the title.
    # When a word is found, perform the next search
    # from the previous words ending position in the
    # title.
    # If anything is skipped (i.e. the search position is 
    # greater the starting index) add whatever's skipped
    # as a word.
    # This method also allows for cases where a title is split
    # in first_page by the texr that, in the rendered PDF, 
    # comes after.
    
    
    p=0
    t=0
    result=""
    
    verbose(first_page)
    
    new_title = ""

    first_page_words = re.sub(r'\(cid:[0-9]+\)', ' ', first_page)
    first_page_words = first_page_words.replace('\n',' ').split(' ')
    title_lower = title_without_space.lower()
    for w in first_page_words:
        if w == '':
            continue
        verbose(f"Searching for {w}")
        pos = title_lower.find(w.lower(), t)
        if pos >= 0 and pos >= t:
            verbose(f"Found at {pos = }")
            # if t > 0 and not new_title.endswith(w + " ")
            if pos > t:
                verbose(f"adding {title_without_space[t:pos] = }")
                new_title += title_without_space[t:pos] + " "
            verbose(f"adding {w = }")
            new_title += w + " "
            verbose(f"{t = }, {pos = }, {len(w) = }, {new_title = }")
            t = pos + len(w)
        if t >= len(title_lower):
            return new_title
    
    return new_title.strip()


def run():
    try:
        parser = argparse.ArgumentParser(
            prog='pdftitle',
            description='Extracts the title of a PDF article',
            epilog='')
        parser.add_argument('-p', '--pdf',
                            help='pdf file', required=True)
        parser.add_argument('-a', '--algo',
                            help='algorithm to derive title, default is ' +
                            'original that finds the text with largest ' +
                            'font size',
                            required=False, default="original")
        parser.add_argument('--replace-missing-char',
                            help='replace missing char with the one ' +
                            'specified')
        parser.add_argument('-c', '--change-name', action='store_true',
                            help='change the name of the pdf file')
        parser.add_argument('-t', '--title-case', action='store_true',
                            help='modify the case of final title to be ' +
                            'title case')
        parser.add_argument('--min_characters', type=int, default=5, \
                            help='minimum number of characters in title')
        parser.add_argument('--min_words', type=int, default=0, \
                            help='minimum number of words in title')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            help='enable verbose logging')

        # Parse aguments and set global parameters
        args = parser.parse_args()
        # pylint: disable=W0603
        global VERBOSE, MISSING_CHAR, ALGO, TITLE_CASE
        VERBOSE = args.verbose
        MISSING_CHAR = args.replace_missing_char
        ALGO = args.algo
        TITLE_CASE = args.title_case
        title = get_title_from_file(args.pdf, min_ch=args.min_characters, min_wd=args.min_words)

        if TITLE_CASE:
            verbose('before title case: %s' % title)
            title = title.title()

        # If no name was found, return a non-zero exit code
        if title is None:
            return 1

        # If the user wants to change the name of the file
        if args.change_name:

            # Change the title to a more pleasant file name
            new_name = title.lower()  # Lower case name
            valid_chars = set(string.ascii_lowercase + string.digits + " ")
            new_name = "".join(c for c in new_name if c in valid_chars)
            new_name = new_name.replace(' ', '_') + ".pdf"

            os.rename(args.pdf, new_name)
            print(new_name)
        else:
            print(title)

        return 0

    except Exception as e:  # pylint: disable=W0612,W0703
        if VERBOSE:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(run())