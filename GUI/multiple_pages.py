import tkinter as tk

root = tk.Tk()
root.geometry('300x400')
root.title('Tkinter Hub')

main_frame = tk.Frame(root)

page_1 = tk.Frame(main_frame)
page_1_lb = tk.Label(page_1, text='Start Page', font=('Bold', 20))
page_1_lb.pack()

page_1.pack(pady=100)

page_2 = tk.Frame(main_frame)
page_2_lb = tk.Label(page_2, text='Home', font=('Bold', 20))
page_2_lb.pack()

page_3 = tk.Frame(main_frame)
page_3_lb = tk.Label(page_3, text='Menu', font=('Bold', 20))
page_3_lb.pack()

page_4 = tk.Frame(main_frame)
page_4_lb = tk.Label(page_4, text='About', font=('Bold', 20))
page_4_lb.pack()


main_frame.pack(fill=tk.BOTH, expand=True)

#pages variable of all the UI
pages = [page_1, page_2, page_3, page_4]
count = 0

def move_next_page():
    global count

    if not count > len(pages) - 2:

     for p in pages:
         p.pack_forget()

    count += 1
    # counts how many pages
    page = pages[count]
    page.pack(pady=100)

def move_back_page():
        global count

        if not count == 0:

            for p in pages:
                p.pack_forget()

        count -= 1
        # counts how many pages
        page = pages[count]
        page.pack(pady=100)



bottom_frame = tk.Frame(root)
# back button UI,  command is using the move back page function
back_btn = tk.Button(bottom_frame, text='Back',
                     font=('Bold', 12),
                     bg='#1877f2', fg='white', width=8,
                     command=move_back_page)
back_btn.pack(side=tk.LEFT, padx=10)
# next button UI, command is using the move next page function
next_btn = tk.Button(bottom_frame, text='Next',
                     font=('Bold', 12),
                     bg='#1877f2', fg='white', width=8,
                     command=move_next_page)
next_btn.pack(side=tk.RIGHT, padx=10)

bottom_frame.pack(side=tk.BOTTOM, pady=10)

root.mainloop()