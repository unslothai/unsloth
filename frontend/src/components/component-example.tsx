import * as React from "react";

import { Example, ExampleWrapper } from "@/components/example";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  Avatar,
  AvatarBadge,
  AvatarFallback,
  AvatarGroup,
  AvatarGroupCount,
  AvatarImage,
} from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuPortal,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import { Progress } from "@/components/ui/progress";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Toggle } from "@/components/ui/toggle";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AlertCircleIcon,
  BluetoothIcon,
  CodeIcon,
  ComputerIcon,
  CreditCardIcon,
  DownloadIcon,
  EyeIcon,
  File01Icon,
  FileIcon,
  FloppyDiskIcon,
  FolderIcon,
  FolderOpenIcon,
  HelpCircleIcon,
  InformationCircleIcon,
  KeyboardIcon,
  LanguageCircleIcon,
  LayoutIcon,
  LogoutIcon,
  MailIcon,
  MoonIcon,
  MoreHorizontalCircle01Icon,
  MoreVerticalCircle01Icon,
  NotificationIcon,
  PaintBoardIcon,
  PanelRightIcon,
  PlusSignIcon,
  SearchIcon,
  SettingsIcon,
  ShieldIcon,
  SunIcon,
  TextBoldIcon,
  TextItalicIcon,
  TextUnderlineIcon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

export function ComponentExample() {
  return (
    <ExampleWrapper>
      <CardExample />
      <FormExample />
      <FormControlsExample />
      <FeedbackExample />
      <DataDisplayExample />
      <NavigationExample />
      <LayoutExample />
      <OverlaysExample />
      <SettingsCardExample />
      <UserTableExample />
      <LoadingStateExample />
    </ExampleWrapper>
  );
}

function CardExample() {
  return (
    <Example title="Card" className="items-center justify-center">
      <Card className="relative w-full max-w-sm overflow-hidden pt-0">
        <div className="bg-primary absolute inset-0 z-30 aspect-video opacity-50 mix-blend-color" />
        <img
          src="https://images.unsplash.com/photo-1604076850742-4c7221f3101b?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
          alt="mymind on Unsplash"
          title="Photo by mymind on Unsplash"
          className="relative z-20 aspect-video w-full object-cover brightness-60 grayscale"
        />
        <CardHeader>
          <CardTitle>Observability Plus is replacing Monitoring</CardTitle>
          <CardDescription>
            Switch to the improved way to explore your data, with natural
            language. Monitoring will no longer be available on the Pro plan in
            November, 2025
          </CardDescription>
        </CardHeader>
        <CardFooter>
          <AlertDialog>
            <AlertDialogTrigger asChild={true}>
              <Button>
                <HugeiconsIcon
                  icon={PlusSignIcon}
                  strokeWidth={2}
                  data-icon="inline-start"
                />
                Show Dialog
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent size="sm">
              <AlertDialogHeader>
                <AlertDialogMedia>
                  <HugeiconsIcon icon={BluetoothIcon} strokeWidth={2} />
                </AlertDialogMedia>
                <AlertDialogTitle>Allow accessory to connect?</AlertDialogTitle>
                <AlertDialogDescription>
                  Do you want to allow the USB accessory to connect to this
                  device?
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Don&apos;t allow</AlertDialogCancel>
                <AlertDialogAction>Allow</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
          <Badge variant="secondary" className="ml-auto">
            Warning
          </Badge>
        </CardFooter>
      </Card>
    </Example>
  );
}

const frameworks = [
  "Next.js",
  "SvelteKit",
  "Nuxt.js",
  "Remix",
  "Astro",
] as const;

function FormExample() {
  const [notifications, setNotifications] = React.useState({
    email: true,
    sms: false,
    push: true,
  });
  const [theme, setTheme] = React.useState("light");

  return (
    <Example title="Form">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>User Information</CardTitle>
          <CardDescription>Please fill in your details below</CardDescription>
          <CardAction>
            <DropdownMenu>
              <DropdownMenuTrigger asChild={true}>
                <Button variant="ghost" size="icon">
                  <HugeiconsIcon
                    icon={MoreVerticalCircle01Icon}
                    strokeWidth={2}
                  />
                  <span className="sr-only">More options</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuGroup>
                  <DropdownMenuLabel>File</DropdownMenuLabel>
                  <DropdownMenuItem>
                    <HugeiconsIcon icon={FileIcon} strokeWidth={2} />
                    New File
                    <DropdownMenuShortcut>⌘N</DropdownMenuShortcut>
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <HugeiconsIcon icon={FolderIcon} strokeWidth={2} />
                    New Folder
                    <DropdownMenuShortcut>⇧⌘N</DropdownMenuShortcut>
                  </DropdownMenuItem>
                  <DropdownMenuSub>
                    <DropdownMenuSubTrigger>
                      <HugeiconsIcon icon={FolderOpenIcon} strokeWidth={2} />
                      Open Recent
                    </DropdownMenuSubTrigger>
                    <DropdownMenuPortal>
                      <DropdownMenuSubContent>
                        <DropdownMenuGroup>
                          <DropdownMenuLabel>Recent Projects</DropdownMenuLabel>
                          <DropdownMenuItem>
                            <HugeiconsIcon icon={CodeIcon} strokeWidth={2} />
                            Project Alpha
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <HugeiconsIcon icon={CodeIcon} strokeWidth={2} />
                            Project Beta
                          </DropdownMenuItem>
                          <DropdownMenuSub>
                            <DropdownMenuSubTrigger>
                              <HugeiconsIcon
                                icon={MoreHorizontalCircle01Icon}
                                strokeWidth={2}
                              />
                              More Projects
                            </DropdownMenuSubTrigger>
                            <DropdownMenuPortal>
                              <DropdownMenuSubContent>
                                <DropdownMenuItem>
                                  <HugeiconsIcon
                                    icon={CodeIcon}
                                    strokeWidth={2}
                                  />
                                  Project Gamma
                                </DropdownMenuItem>
                                <DropdownMenuItem>
                                  <HugeiconsIcon
                                    icon={CodeIcon}
                                    strokeWidth={2}
                                  />
                                  Project Delta
                                </DropdownMenuItem>
                              </DropdownMenuSubContent>
                            </DropdownMenuPortal>
                          </DropdownMenuSub>
                        </DropdownMenuGroup>
                        <DropdownMenuSeparator />
                        <DropdownMenuGroup>
                          <DropdownMenuItem>
                            <HugeiconsIcon icon={SearchIcon} strokeWidth={2} />
                            Browse...
                          </DropdownMenuItem>
                        </DropdownMenuGroup>
                      </DropdownMenuSubContent>
                    </DropdownMenuPortal>
                  </DropdownMenuSub>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>
                    <HugeiconsIcon icon={FloppyDiskIcon} strokeWidth={2} />
                    Save
                    <DropdownMenuShortcut>⌘S</DropdownMenuShortcut>
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <HugeiconsIcon icon={DownloadIcon} strokeWidth={2} />
                    Export
                    <DropdownMenuShortcut>⇧⌘E</DropdownMenuShortcut>
                  </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                  <DropdownMenuLabel>View</DropdownMenuLabel>
                  <DropdownMenuCheckboxItem
                    checked={notifications.email}
                    onCheckedChange={(checked) =>
                      setNotifications({
                        ...notifications,
                        email: checked === true,
                      })
                    }
                  >
                    <HugeiconsIcon icon={EyeIcon} strokeWidth={2} />
                    Show Sidebar
                  </DropdownMenuCheckboxItem>
                  <DropdownMenuCheckboxItem
                    checked={notifications.sms}
                    onCheckedChange={(checked) =>
                      setNotifications({
                        ...notifications,
                        sms: checked === true,
                      })
                    }
                  >
                    <HugeiconsIcon icon={LayoutIcon} strokeWidth={2} />
                    Show Status Bar
                  </DropdownMenuCheckboxItem>
                  <DropdownMenuSub>
                    <DropdownMenuSubTrigger>
                      <HugeiconsIcon icon={PaintBoardIcon} strokeWidth={2} />
                      Theme
                    </DropdownMenuSubTrigger>
                    <DropdownMenuPortal>
                      <DropdownMenuSubContent>
                        <DropdownMenuGroup>
                          <DropdownMenuLabel>Appearance</DropdownMenuLabel>
                          <DropdownMenuRadioGroup
                            value={theme}
                            onValueChange={setTheme}
                          >
                            <DropdownMenuRadioItem value="light">
                              <HugeiconsIcon icon={SunIcon} strokeWidth={2} />
                              Light
                            </DropdownMenuRadioItem>
                            <DropdownMenuRadioItem value="dark">
                              <HugeiconsIcon icon={MoonIcon} strokeWidth={2} />
                              Dark
                            </DropdownMenuRadioItem>
                            <DropdownMenuRadioItem value="system">
                              <HugeiconsIcon
                                icon={ComputerIcon}
                                strokeWidth={2}
                              />
                              System
                            </DropdownMenuRadioItem>
                          </DropdownMenuRadioGroup>
                        </DropdownMenuGroup>
                      </DropdownMenuSubContent>
                    </DropdownMenuPortal>
                  </DropdownMenuSub>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                  <DropdownMenuLabel>Account</DropdownMenuLabel>
                  <DropdownMenuItem>
                    <HugeiconsIcon icon={UserIcon} strokeWidth={2} />
                    Profile
                    <DropdownMenuShortcut>⇧⌘P</DropdownMenuShortcut>
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <HugeiconsIcon icon={CreditCardIcon} strokeWidth={2} />
                    Billing
                  </DropdownMenuItem>
                  <DropdownMenuSub>
                    <DropdownMenuSubTrigger>
                      <HugeiconsIcon icon={SettingsIcon} strokeWidth={2} />
                      Settings
                    </DropdownMenuSubTrigger>
                    <DropdownMenuPortal>
                      <DropdownMenuSubContent>
                        <DropdownMenuGroup>
                          <DropdownMenuLabel>Preferences</DropdownMenuLabel>
                          <DropdownMenuItem>
                            <HugeiconsIcon
                              icon={KeyboardIcon}
                              strokeWidth={2}
                            />
                            Keyboard Shortcuts
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <HugeiconsIcon
                              icon={LanguageCircleIcon}
                              strokeWidth={2}
                            />
                            Language
                          </DropdownMenuItem>
                          <DropdownMenuSub>
                            <DropdownMenuSubTrigger>
                              <HugeiconsIcon
                                icon={NotificationIcon}
                                strokeWidth={2}
                              />
                              Notifications
                            </DropdownMenuSubTrigger>
                            <DropdownMenuPortal>
                              <DropdownMenuSubContent>
                                <DropdownMenuGroup>
                                  <DropdownMenuLabel>
                                    Notification Types
                                  </DropdownMenuLabel>
                                  <DropdownMenuCheckboxItem
                                    checked={notifications.push}
                                    onCheckedChange={(checked) =>
                                      setNotifications({
                                        ...notifications,
                                        push: checked === true,
                                      })
                                    }
                                  >
                                    <HugeiconsIcon
                                      icon={NotificationIcon}
                                      strokeWidth={2}
                                    />
                                    Push Notifications
                                  </DropdownMenuCheckboxItem>
                                  <DropdownMenuCheckboxItem
                                    checked={notifications.email}
                                    onCheckedChange={(checked) =>
                                      setNotifications({
                                        ...notifications,
                                        email: checked === true,
                                      })
                                    }
                                  >
                                    <HugeiconsIcon
                                      icon={MailIcon}
                                      strokeWidth={2}
                                    />
                                    Email Notifications
                                  </DropdownMenuCheckboxItem>
                                </DropdownMenuGroup>
                              </DropdownMenuSubContent>
                            </DropdownMenuPortal>
                          </DropdownMenuSub>
                        </DropdownMenuGroup>
                        <DropdownMenuSeparator />
                        <DropdownMenuGroup>
                          <DropdownMenuItem>
                            <HugeiconsIcon icon={ShieldIcon} strokeWidth={2} />
                            Privacy & Security
                          </DropdownMenuItem>
                        </DropdownMenuGroup>
                      </DropdownMenuSubContent>
                    </DropdownMenuPortal>
                  </DropdownMenuSub>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                  <DropdownMenuItem>
                    <HugeiconsIcon icon={HelpCircleIcon} strokeWidth={2} />
                    Help & Support
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <HugeiconsIcon icon={File01Icon} strokeWidth={2} />
                    Documentation
                  </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                  <DropdownMenuItem variant="destructive">
                    <HugeiconsIcon icon={LogoutIcon} strokeWidth={2} />
                    Sign Out
                    <DropdownMenuShortcut>⇧⌘Q</DropdownMenuShortcut>
                  </DropdownMenuItem>
                </DropdownMenuGroup>
              </DropdownMenuContent>
            </DropdownMenu>
          </CardAction>
        </CardHeader>
        <CardContent>
          <form>
            <FieldGroup>
              <div className="grid grid-cols-2 gap-4">
                <Field>
                  <FieldLabel htmlFor="small-form-name">Name</FieldLabel>
                  <Input
                    id="small-form-name"
                    placeholder="Enter your name"
                    required={true}
                    className=""
                  />
                </Field>
                <Field>
                  <FieldLabel htmlFor="small-form-role">Role</FieldLabel>
                  <Select defaultValue="">
                    <SelectTrigger id="small-form-role">
                      <SelectValue placeholder="Select a role" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="developer">Developer</SelectItem>
                        <SelectItem value="designer">Designer</SelectItem>
                        <SelectItem value="manager">Manager</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </Field>
              </div>
              <Field>
                <FieldLabel htmlFor="small-form-framework">
                  Framework
                </FieldLabel>
                <Combobox items={frameworks}>
                  <ComboboxInput
                    id="small-form-framework"
                    placeholder="Select a framework"
                    required={true}
                  />
                  <ComboboxContent>
                    <ComboboxEmpty>No frameworks found.</ComboboxEmpty>
                    <ComboboxList>
                      {(item) => (
                        <ComboboxItem key={item} value={item}>
                          {item}
                        </ComboboxItem>
                      )}
                    </ComboboxList>
                  </ComboboxContent>
                </Combobox>
              </Field>
              <Field>
                <FieldLabel htmlFor="small-form-comments">Comments</FieldLabel>
                <Textarea
                  id="small-form-comments"
                  placeholder="Add any additional comments"
                />
              </Field>
              <Field orientation="horizontal">
                <Button type="submit">Submit</Button>
                <Button variant="outline" type="button">
                  Cancel
                </Button>
              </Field>
            </FieldGroup>
          </form>
        </CardContent>
      </Card>
    </Example>
  );
}

function FormControlsExample() {
  const [checked, setChecked] = React.useState(true);
  const [radioValue, setRadioValue] = React.useState("option-1");
  const [switchOn, setSwitchOn] = React.useState(true);
  const [sliderValue, setSliderValue] = React.useState([50]);
  const [rangeValue, setRangeValue] = React.useState([25, 75]);

  return (
    <Example title="Form Controls">
      <div className="grid w-full gap-6">
        {/* Checkbox */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Checkbox</Label>
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <Checkbox
                id="terms"
                checked={checked}
                onCheckedChange={(c) => setChecked(c === true)}
              />
              <Label htmlFor="terms" className="font-normal">
                Accept terms and conditions
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="marketing" />
              <Label htmlFor="marketing" className="font-normal">
                Receive marketing emails
              </Label>
            </div>
          </div>
        </div>

        {/* RadioGroup */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Radio Group</Label>
          <RadioGroup value={radioValue} onValueChange={setRadioValue}>
            <div className="flex items-center gap-2">
              <RadioGroupItem value="option-1" id="option-1" />
              <Label htmlFor="option-1" className="font-normal">
                Default
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <RadioGroupItem value="option-2" id="option-2" />
              <Label htmlFor="option-2" className="font-normal">
                Comfortable
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <RadioGroupItem value="option-3" id="option-3" />
              <Label htmlFor="option-3" className="font-normal">
                Compact
              </Label>
            </div>
          </RadioGroup>
        </div>

        {/* Switch */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Switch</Label>
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <Switch
                id="airplane"
                checked={switchOn}
                onCheckedChange={setSwitchOn}
              />
              <Label htmlFor="airplane" className="font-normal">
                Airplane Mode
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Switch id="small-switch" size="sm" />
              <Label htmlFor="small-switch" className="font-normal">
                Small Switch
              </Label>
            </div>
          </div>
        </div>

        {/* Slider */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Slider</Label>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="font-normal">Single: {sliderValue[0]}%</Label>
              <Slider
                value={sliderValue}
                onValueChange={setSliderValue}
                max={100}
                step={1}
              />
            </div>
            <div className="space-y-2">
              <Label className="font-normal">
                Range: {rangeValue[0]} - {rangeValue[1]}
              </Label>
              <Slider
                value={rangeValue}
                onValueChange={setRangeValue}
                max={100}
                step={1}
              />
            </div>
          </div>
        </div>

        {/* Toggle Group */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Toggle Group</Label>
          <div className="flex flex-col gap-3">
            <ToggleGroup type="multiple" variant="outline">
              <ToggleGroupItem value="bold" aria-label="Toggle bold">
                <HugeiconsIcon icon={TextBoldIcon} strokeWidth={2} />
              </ToggleGroupItem>
              <ToggleGroupItem value="italic" aria-label="Toggle italic">
                <HugeiconsIcon icon={TextItalicIcon} strokeWidth={2} />
              </ToggleGroupItem>
              <ToggleGroupItem value="underline" aria-label="Toggle underline">
                <HugeiconsIcon icon={TextUnderlineIcon} strokeWidth={2} />
              </ToggleGroupItem>
            </ToggleGroup>
            <Toggle variant="outline" aria-label="Toggle italic">
              <HugeiconsIcon icon={TextItalicIcon} strokeWidth={2} />
              Italic
            </Toggle>
          </div>
        </div>
      </div>
    </Example>
  );
}

function FeedbackExample() {
  return (
    <Example title="Feedback">
      <div className="grid w-full gap-6">
        {/* Alert */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Alert</Label>
          <div className="space-y-3">
            <Alert className="!corner-squircle rounded-[50px]">
              <HugeiconsIcon icon={InformationCircleIcon} strokeWidth={2} />
              <AlertTitle>Heads up!</AlertTitle>
              <AlertDescription>
                You can add components to your app using the CLI.
              </AlertDescription>
            </Alert>
            <Alert variant="destructive" className="rounded-[20px]">
              <HugeiconsIcon icon={AlertCircleIcon} strokeWidth={2} />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>
                Your session has expired. Please log in again.
              </AlertDescription>
            </Alert>
          </div>
        </div>

        {/* Progress */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Progress</Label>
          <div className="space-y-3">
            <div className="space-y-1">
              <Label className="text-xs font-normal">25%</Label>
              <Progress value={25} />
            </div>
            <div className="space-y-1">
              <Label className="text-xs font-normal">50%</Label>
              <Progress value={50} />
            </div>
            <div className="space-y-1">
              <Label className="text-xs font-normal">75%</Label>
              <Progress value={75} />
            </div>
          </div>
        </div>

        {/* Skeleton */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Skeleton</Label>
          <div className="flex items-center space-x-4">
            <Skeleton className="h-12 w-12 rounded-full" />
            <div className="space-y-2">
              <Skeleton className="h-4 w-[250px]" />
              <Skeleton className="h-4 w-[200px]" />
            </div>
          </div>
        </div>

        {/* Tooltip */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Tooltip</Label>
          <div className="flex gap-2">
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <Button variant="outline">Hover me</Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>This is a tooltip</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
      </div>
    </Example>
  );
}

function DataDisplayExample() {
  return (
    <Example title="Data Display">
      <div className="grid w-full gap-6">
        {/* Avatar */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Avatar</Label>
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-4">
              <Avatar>
                <AvatarImage
                  src="https://github.com/shadcn.png"
                  alt="@shadcn"
                />
                <AvatarFallback>CN</AvatarFallback>
              </Avatar>
              <Avatar>
                <AvatarImage
                  src="https://github.com/vercel.png"
                  alt="@vercel"
                />
                <AvatarFallback>VC</AvatarFallback>
                <AvatarBadge />
              </Avatar>
              <Avatar size="lg">
                <AvatarFallback>JD</AvatarFallback>
              </Avatar>
            </div>
            <AvatarGroup>
              <Avatar>
                <AvatarImage
                  src="https://github.com/shadcn.png"
                  alt="@shadcn"
                />
                <AvatarFallback>CN</AvatarFallback>
              </Avatar>
              <Avatar>
                <AvatarImage
                  src="https://github.com/vercel.png"
                  alt="@vercel"
                />
                <AvatarFallback>VC</AvatarFallback>
              </Avatar>
              <Avatar>
                <AvatarFallback>JD</AvatarFallback>
              </Avatar>
              <AvatarGroupCount>+3</AvatarGroupCount>
            </AvatarGroup>
          </div>
        </div>

        {/* Badge */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Badge</Label>
          <div className="flex flex-wrap gap-2">
            <Badge>Default</Badge>
            <Badge variant="secondary">Secondary</Badge>
            <Badge variant="outline">Outline</Badge>
            <Badge variant="destructive">Destructive</Badge>
          </div>
        </div>

        {/* Table */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Table</Label>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Role</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell>John Doe</TableCell>
                <TableCell>
                  <Badge variant="secondary">Active</Badge>
                </TableCell>
                <TableCell>Admin</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Jane Smith</TableCell>
                <TableCell>
                  <Badge variant="outline">Pending</Badge>
                </TableCell>
                <TableCell>User</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Bob Wilson</TableCell>
                <TableCell>
                  <Badge variant="destructive">Inactive</Badge>
                </TableCell>
                <TableCell>User</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>
      </div>
    </Example>
  );
}

function NavigationExample() {
  return (
    <Example title="Navigation">
      <div className="grid w-full gap-6">
        {/* Tabs */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Tabs</Label>
          <div className="space-y-4">
            <Tabs defaultValue="account">
              <TabsList>
                <TabsTrigger value="account">Account</TabsTrigger>
                <TabsTrigger value="password">Password</TabsTrigger>
                <TabsTrigger value="settings">Settings</TabsTrigger>
              </TabsList>
              <TabsContent value="account">
                Account settings content.
              </TabsContent>
              <TabsContent value="password">
                Password settings content.
              </TabsContent>
              <TabsContent value="settings">
                General settings content.
              </TabsContent>
            </Tabs>
            <Tabs defaultValue="overview">
              <TabsList variant="line">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="analytics">Analytics</TabsTrigger>
                <TabsTrigger value="reports">Reports</TabsTrigger>
              </TabsList>
              <TabsContent value="overview">Overview content.</TabsContent>
              <TabsContent value="analytics">Analytics content.</TabsContent>
              <TabsContent value="reports">Reports content.</TabsContent>
            </Tabs>
          </div>
        </div>

        {/* Breadcrumb */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Breadcrumb</Label>
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbLink href="#">Home</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbLink href="#">Components</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbPage>Breadcrumb</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </div>

        {/* Pagination */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Pagination</Label>
          <Pagination>
            <PaginationContent>
              <PaginationItem>
                <PaginationPrevious href="#" />
              </PaginationItem>
              <PaginationItem>
                <PaginationLink href="#">1</PaginationLink>
              </PaginationItem>
              <PaginationItem>
                <PaginationLink href="#" isActive={true}>
                  2
                </PaginationLink>
              </PaginationItem>
              <PaginationItem>
                <PaginationLink href="#">3</PaginationLink>
              </PaginationItem>
              <PaginationItem>
                <PaginationNext href="#" />
              </PaginationItem>
            </PaginationContent>
          </Pagination>
        </div>
      </div>
    </Example>
  );
}

function LayoutExample() {
  return (
    <Example title="Layout">
      <div className="grid w-full gap-6">
        {/* Accordion */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Accordion</Label>
          <Accordion type="single" collapsible={true}>
            <AccordionItem value="item-1">
              <AccordionTrigger>Is it accessible?</AccordionTrigger>
              <AccordionContent>
                Yes. It adheres to the WAI-ARIA design pattern.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-2">
              <AccordionTrigger>Is it styled?</AccordionTrigger>
              <AccordionContent>
                Yes. It comes with default styles that matches the other
                components.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-3">
              <AccordionTrigger>Is it animated?</AccordionTrigger>
              <AccordionContent>
                Yes. It&apos;s animated by default, but you can disable it if
                you prefer.
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      </div>
    </Example>
  );
}

function OverlaysExample() {
  return (
    <Example title="Overlays">
      <div className="grid w-full gap-6">
        {/* Dialog */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Dialog</Label>
          <Dialog>
            <DialogTrigger asChild={true}>
              <Button variant="outline">Open Dialog</Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Edit profile</DialogTitle>
                <DialogDescription>
                  Make changes to your profile here. Click save when you&apos;re
                  done.
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <Field>
                  <FieldLabel htmlFor="name">Name</FieldLabel>
                  <Input id="name" defaultValue="Pedro Duarte" />
                </Field>
                <Field>
                  <FieldLabel htmlFor="username">Username</FieldLabel>
                  <Input id="username" defaultValue="@peduarte" />
                </Field>
              </div>
              <DialogFooter>
                <Button type="submit">Save changes</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>

        {/* Sheet */}
        <div className="space-y-3">
          <Label className="text-muted-foreground text-xs">Sheet</Label>
          <Sheet>
            <SheetTrigger asChild={true}>
              <Button variant="outline">
                <HugeiconsIcon
                  icon={PanelRightIcon}
                  strokeWidth={2}
                  data-icon="inline-start"
                />
                Open Sheet
              </Button>
            </SheetTrigger>
            <SheetContent>
              <SheetHeader>
                <SheetTitle>Edit profile</SheetTitle>
                <SheetDescription>
                  Make changes to your profile here. Click save when you&apos;re
                  done.
                </SheetDescription>
              </SheetHeader>
              <div className="grid gap-4 p-6">
                <Field>
                  <FieldLabel htmlFor="sheet-name">Name</FieldLabel>
                  <Input id="sheet-name" defaultValue="Pedro Duarte" />
                </Field>
                <Field>
                  <FieldLabel htmlFor="sheet-username">Username</FieldLabel>
                  <Input id="sheet-username" defaultValue="@peduarte" />
                </Field>
              </div>
              <SheetFooter>
                <Button type="submit">Save changes</Button>
              </SheetFooter>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </Example>
  );
}

function SettingsCardExample() {
  const [emailNotifications, setEmailNotifications] = React.useState(true);
  const [pushNotifications, setPushNotifications] = React.useState(false);
  const [marketingEmails, setMarketingEmails] = React.useState(false);

  return (
    <Example title="Settings Card">
      <Card className="w-full  max-w-md">
        <CardHeader>
          <CardTitle>Notifications</CardTitle>
          <CardDescription>
            Configure how you receive notifications.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="email-notif">Email notifications</Label>
              <p className="text-muted-foreground text-xs">
                Receive emails about your account.
              </p>
            </div>
            <Switch
              id="email-notif"
              checked={emailNotifications}
              onCheckedChange={setEmailNotifications}
            />
          </div>
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="push-notif">Push notifications</Label>
              <p className="text-muted-foreground text-xs">
                Receive push notifications on your device.
              </p>
            </div>
            <Switch
              id="push-notif"
              checked={pushNotifications}
              onCheckedChange={setPushNotifications}
            />
          </div>
          <div className="flex items-center gap-2 pt-2">
            <Checkbox
              id="marketing"
              checked={marketingEmails}
              onCheckedChange={(c) => setMarketingEmails(c === true)}
            />
            <Label htmlFor="marketing" className="font-normal">
              Receive marketing emails
            </Label>
          </div>
        </CardContent>
        <CardFooter>
          <Button className="w-full">Save preferences</Button>
        </CardFooter>
      </Card>
    </Example>
  );
}

function UserTableExample() {
  const [selectedUsers, setSelectedUsers] = React.useState<string[]>(["1"]);

  const toggleUser = (id: string) => {
    setSelectedUsers((prev) =>
      prev.includes(id) ? prev.filter((u) => u !== id) : [...prev, id],
    );
  };

  const users = [
    {
      id: "1",
      name: "Olivia Martin",
      email: "olivia@example.com",
      status: "active",
      avatar: "https://github.com/shadcn.png",
    },
    {
      id: "2",
      name: "Jackson Lee",
      email: "jackson@example.com",
      status: "pending",
      avatar: "https://github.com/vercel.png",
    },
    {
      id: "3",
      name: "Isabella Nguyen",
      email: "isabella@example.com",
      status: "inactive",
      avatar: "",
    },
  ];

  return (
    <Example title="User Table">
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Team Members</CardTitle>
          <CardDescription>
            Manage your team members and their roles.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-12" />
                <TableHead>User</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {users.map((user) => (
                <TableRow key={user.id}>
                  <TableCell>
                    <Checkbox
                      checked={selectedUsers.includes(user.id)}
                      onCheckedChange={() => toggleUser(user.id)}
                    />
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-3">
                      <Avatar size="sm">
                        {user.avatar && (
                          <AvatarImage src={user.avatar} alt={user.name} />
                        )}
                        <AvatarFallback>
                          {user.name
                            .split(" ")
                            .map((n) => n[0])
                            .join("")}
                        </AvatarFallback>
                      </Avatar>
                      <div>
                        <div className="font-medium">{user.name}</div>
                        <div className="text-muted-foreground text-xs">
                          {user.email}
                        </div>
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        user.status === "active"
                          ? "secondary"
                          : user.status === "pending"
                            ? "outline"
                            : "destructive"
                      }
                    >
                      {user.status}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </Example>
  );
}

function LoadingStateExample() {
  return (
    <Example title="Loading State">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex items-center gap-4">
            <Skeleton className="h-12 w-12 rounded-full" />
            <div className="space-y-2">
              <Skeleton className="h-4 w-[150px]" />
              <Skeleton className="h-3 w-[100px]" />
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <div className="pt-2">
            <Skeleton className="h-32 w-full rounded-xl" />
          </div>
        </CardContent>
        <CardFooter className="gap-2">
          <Skeleton className="h-9 w-24 rounded-4xl" />
          <Skeleton className="h-9 w-24 rounded-4xl" />
        </CardFooter>
      </Card>
    </Example>
  );
}
